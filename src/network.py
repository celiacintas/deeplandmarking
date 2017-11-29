import numpy as np
import theano
from nolearn.lasagne import BatchIterator
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score, r2_score
import matplotlib.pyplot as plt


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class FlipBatchIterator(BatchIterator):

    def transform(self, xb, yb):
        xb, yb = super(FlipBatchIterator, self).transform(xb, yb)
        # TODO use rotate from scipy
        bs = xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)  # /2 choose all
        xb[indices] = xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

        return xb, yb


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class CNN(object):
    def __init__(self, *args):
        try:
            self.args = dict(args)
        except TypeError:
            print("load args as tuples")

    def create_custom_net(self, layers):
        pass

    def create_best_net(self):
        # TODO crossval till we have the RL
        pass

    def training(self):
        pass

    def plot_loss(self, x_label='epoch', y_label='loss',
                  out='/tmp/loss_plot.png'):
        plt.clf()
        train_loss = np.array([i["train_loss"] for i in self.cnn_net.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in self.cnn_net.train_history_])
        plt.plot(train_loss, linewidth=3, label="train")
        plt.plot(valid_loss, linewidth=3, label="valid")
        plt.grid()
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.yscale("log")
        plt.savefig(out)

    def plot_samples(self, y_pred, ids=None, out_folder='/tmp/'):
        # TODO fix this to put names when predict stage
        for i in range(y_pred.shape[0]):
            plt.clf()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
            img = x[i].reshape(self.image_size)
            ax.imshow(img, cmap='gray')
            # TODO apply minmax scaler before plot
            ax.scatter(y_pred[i][0::2], y_pred[i][1::2], marker='x', s=5)
            plt.savefig(out_folder + 'lateral_{}.png'.format(str(i)))

    def report_metrics(self, X_test, y_test):
        # TODO plots should be at other place
        self.plot_loss(self.cnn_net)
        y_pred = self.cnn_net.predict(X_test)
        self.plot_sample(X_test, y_pred, names=None)
        # TODO this should be over the minmax scale values
        rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred.ravel()))
        ev = explained_variance_score(y_test.ravel(), y_pred.ravel())
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return rmse, ev, mae, r2
