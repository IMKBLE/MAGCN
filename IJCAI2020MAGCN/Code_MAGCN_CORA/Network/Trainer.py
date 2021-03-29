import tensorflow as tf
from Network.Graph_Attention_Encoder import GATE
from utils import process
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
import numpy as np

# 导入计算指标MNI,ARI
nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    # labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def f_score(labels_true, labels_pred):
    """Compute the B^3 variant of F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float f_score: calculated F-score
    """
    _, _, f = b3_precision_recall_fscore(labels_true, labels_pred)
    return f

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class Trainer():
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        self.gate = GATE(args.hidden_dims1, args.hidden_dims2, args.lambda_)
        self.loss, self.H, self.C, self.H2, self.C2, self.pred, self.dense_loss, self.z, self.z2, self.features_loss, self.structure_loss = self.gate(self.A, self.X, self.R, self.S, self.p, self.A2, self.X2, self.R2, self.S2)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)
        self.A2 = tf.sparse_placeholder(dtype=tf.float32)
        self.X2 = tf.placeholder(dtype=tf.float32)
        self.S2 = tf.placeholder(tf.int64)
        self.R2 = tf.placeholder(tf.int64)
        self.p = tf.placeholder(tf.float32, shape=(None, 7))

    def build_session(self, gpu=True):
        # 设置GPU按需增长
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not gpu:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        # print(variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, X, S, R, A2, X2, S2, R2, L, fin=False):
        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch, A, X, S, R, A2, X2, S2, R2, L, fin)

    def run_epoch(self, epoch, A, X, S, R, A2, X2, S2, R2, L, fin):
        # print('self.gate.q:' + str(self.gate.q.shape))
        # print('self.S2:' + str(self.S2.shape))
        q = self.session.run(self.gate.q, feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2})
        p = self.gate.target_distribution(q)
        if not fin:
            loss, pred, _, st_loss, f_loss, d_loss = self.session.run([self.loss, self.pred, self.train_op, self.structure_loss, self.features_loss, self.dense_loss],
                                             feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.p: p, self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2})
            if epoch % 5 == 0:
                print(
                    "Epoch--{}:\tloss: {:.8f}\t\tsloss: {:.8f}\t\tfloss: {:.8f}\t\tdloss: {:.8f}\t\nacc: {:.8f}\t\tnmi: {:.8f}\t\tf1: {:.8f}\t\tari: {:.8f}".
                    format(epoch, loss, st_loss, f_loss, d_loss, cluster_acc(L, pred), nmi(L, pred), f_score(L, pred),
                           ari(L, pred)))
        elif fin:
            loss, pred, _, st_loss, f_loss, d_loss = self.session.run([self.loss, self.pred, self.train_op, self.structure_loss, self.features_loss, self.dense_loss],
                                             feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.p: p, self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2})

            # g_loss, _ = self.session.run([self.g_loss, self.Gtrain_op],
            #                              feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.p: p})
            # pred = self.session.run(self.pred,
            #                         feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.p: p})
            # print("Epoch: %s, Loss: %.5f" % (epoch, loss))
            if epoch % 5 == 0:
                print("Epoch--{}:\tloss: {:.8f}\t\tsloss: {:.8f}\t\tfloss: {:.8f}\t\tdloss: {:.8f}\t\nacc: {:.8f}\t\tnmi: {:.8f}\t\tf1: {:.8f}\t\tari: {:.8f}".
                    format(epoch, loss, st_loss, f_loss, d_loss, cluster_acc(L, pred), nmi(L, pred), f_score(L, pred), ari(L, pred)))

    def infer(self, A, X, S, R, A2, X2, S2, R2):
        H, C, z, H2, C2, z2 = self.session.run([self.H, self.C, self.z, self.H2, self.C2, self.z2],
                                feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2})
        return H, process.conver_sparse_tf2np(C), z, H2, process.conver_sparse_tf2np(C2), z2

    def assign(self, A, X, S, R, A2, X2, S2, R2):
        _, _, embeddings, _, _, embeddings2 = self.infer(A, X, S, R, A2, X2, S2, R2)
        assign_mu_op = self.gate.get_assign_cluster_centers_op((1.0 * embeddings + 0 * embeddings2))
        _ = self.session.run(assign_mu_op)