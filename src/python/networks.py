import numpy as np

np.set_printoptions(suppress=True)
import tensorflow as tf


class InvariantModule(tf.keras.Model):
    """Implements an invariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, meta, pooler=tf.reduce_mean):
        """
        Creates an invariant function with mean pooling.
        ----------

        Arguments:
        model_settings : dict -- hyperparameter settings for the invariant module
        """

        super(InvariantModule, self).__init__()

        # Pre pooling network
        self.pre_pooling_dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(**meta["dense_inv_pre_pooling_args"])
                for _ in range(meta["n_dense_inv"])
            ]
        )

        self.pooler = pooler

        # Post pooling network
        self.post_pooling_dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(**meta["dense_inv_post_pooling_args"])
                for _ in range(meta["n_dense_inv"])
            ]
        )

    def call(self, x):
        """
        Transforms the input into an invariant representation.
        ----------

        Arguments:
        x : tf.Tensor of variable shape - 4-dimensional with (batch_size, n_clusters, n_obs, n_variables)
                                          or 3-dimensional with (batch_size, n_clusters, inv_embedding)
        --------

        Returns:
        out: tf.Tensor of variable shape - either 3-dimensional (when input is 4D) with (batch_size, n_clusters, inv_embedding)
                                           or 2-dimensional (when input is 3D) with (batch_size, inv_embedding)
        """

        # Embed input before pooling
        x_emb = self.pre_pooling_dense(x)

        # Perform mean pooling, shape of pooled is (batch_size, K, dense_out)
        pooled = self.pooler(
            x_emb, axis=-2
        )  # always reduce dimensionality of the lowest exchangable data level

        # Increase representational power
        out = self.post_pooling_dense(pooled)
        return out


class EquivariantModule(tf.keras.Model):
    def __init__(self, meta):
        """
        Creates an equivariant function.
        ----------

        Arguments:
        model_settings : dict -- hyperparameter settings for the equivariant module
        """

        super(EquivariantModule, self).__init__()

        self.inv = InvariantModule(meta["inv_inner"])
        self.equiv = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(**meta["dense_equiv_args"])
                for _ in range(meta["n_dense_equiv"])
            ]
        )

    def call(self, x):
        """
        Performs an equivariant transformation using both equiv and inv transforms.
        """

        n_rep = x.shape[-2]
        inv = self.inv(x)
        inv = tf.stack([inv] * n_rep, axis=-2)
        inp = tf.concat([x, inv], axis=-1)
        out = self.equiv(inp)
        return out


class InvariantNetwork(tf.keras.Model):
    def __init__(self, meta, pooler=tf.reduce_mean):
        """
        Creates an invariant network consisting of equivariant and invariant functions.
        ----------

        Arguments:
        model_settings : dict -- hyperparameter settings for the invariant network
        """

        super(InvariantNetwork, self).__init__()

        self.inv = InvariantModule(meta["inv_outer"])
        self.equiv = tf.keras.Sequential(
            [EquivariantModule(meta) for _ in range(meta["n_equiv"])]
        )

    def call(self, x):
        """
        Performs an invariant transformation using both equiv and inv transforms.
        """

        out = self.equiv(x)
        out = self.inv(out)
        return out


class HierarchicalInvariantNetwork(tf.keras.Model):
    """
    Implements a network that is able to summarize hierarchical data.
    """

    def __init__(self, meta):
        """
        Creates a hierarchical Network consisting of two stacked invariant modules.
        ----------

        Arguments:
        model_settings : dict -- hyperparameter settings for the hierarchical network
        """

        super(HierarchicalInvariantNetwork, self).__init__()

        self.inv_4d = InvariantNetwork(meta["level_1"])
        self.inv_3d = InvariantNetwork(meta["level_2"])

    def call(self, x):
        """
        Transforms the 4-dimensional input into learned summary statistics.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n_clusters, n_obs, n_variables) -- simulated data
        --------

        Returns:
        out : tf.Tensor of shape (batch_size, (dense_inv_post_pooling_args on level_2) + 2) -- the learned summary statistics
        """

        # Pass through invariant networks
        out = self.inv_4d(x)
        out = self.inv_3d(out)

        # Extract number of clusters and observations into a repeated vector and concatenate with network output
        n_clust = int(x.shape[-3])
        n_clust_rep = tf.math.sqrt(n_clust * tf.ones((x.shape[0], 1)))
        out = tf.concat((out, n_clust_rep), axis=-1)

        n_obs = int(x.shape[-2])
        n_obs_rep = tf.math.sqrt(n_obs * tf.ones((x.shape[0], 1)))
        out = tf.concat((out, n_obs_rep), axis=-1)

        return out


class ModelProbabilityNetwork(tf.keras.Model):
    """
    Implements a network that infers the posterior probabilities of competing statistical models.
    """

    def __init__(self, meta):
        """
        Creates an model probability network.
        ----------

        Arguments:
        meta : dict -- hyperparameter settings for the model probability network
        """

        super(ModelProbabilityNetwork, self).__init__()

        # Hidden layers
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(**meta["dense_args"])
                for _ in range(meta["n_dense"])
            ]
        )

        # Output layer
        self.softmax_layer = tf.keras.layers.Dense(
            meta["n_models"], activation="softmax"
        )

        # Output layer without activation for obtaining logits
        self.logit_layer = tf.keras.layers.Dense(meta["n_models"])

        self.J = meta["n_models"]

    def call(self, x):
        """
        Forward pass through the model probability network that returns approximated model probabilities.
        """

        out = self.dense(x)
        probs = self.softmax_layer(out)

        return probs

    def predict(self, obs_data, to_numpy=True):
        """
        Returns a dictionary with approximated model probabilities that are optionally in numpy format.

        Parameters
        ----------
        obs_data: tf.Tensor
            Observed data or respective embeddings created by a summary network
        to_numpy: bool, default: True
            Flag that controls whether the output is a np.array or tf.Tensor

        Returns
        -------
        out: dict
            Dictionary with key {m_probs}
        """

        probs = self(obs_data)

        if to_numpy:
            probs = probs.numpy()

        return {"m_probs": probs}

    def get_logits(self, obs_data):
        """Shortcut function to obtain logits.

        Parameters
        ----------
        obs_data: tf.Tensor
            Observed data or respective embeddings created by a summary network

        Returns
        -------
        logits: tf.Tensor of shape (batch_size, num_models)
            The logits for each class
        """

        out = self.dense(obs_data)
        logits = self.logit_layer(out)

        return logits
