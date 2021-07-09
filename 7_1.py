from scipy.stats import multivariate_normal
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

class GaussianMixture0:
    def __init__(self, n_components: int = 1, covariance_type: str = 'full',
                 tol: float = 0.001, reg_covar: float = 1e-06, max_iter: int = 100):
        self.n_components=n_components
        self.means_=None#均值
        self.covariances_=None#协方差
        self.weights_=None#各模型的权值
        self.reg_covar=reg_covar#为了防止出现奇异值矩阵
        self.max_iter=max_iter

    def Gaussian(self,x,mean,cov):
        dim=np.shape(cov)[0]
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1, dim))
        # 概率密度
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob


    def fit(self,X_train):
        n_samples, n_feature = X_train.shape
        self.reg_covar = self.reg_covar * np.identity(n_feature)

        # 初始化一些必要的参数：均值，协方差，权重
        self.means_ = np.random.randint(X_train.min() / 2, X_train.max() /
                                        2, size=(self.n_components, n_feature))

        self.covariances_ = np.zeros((self.n_components, n_feature, n_feature))
        for k in range(self.n_components):
            np.fill_diagonal(self.covariances_[k], 1)

        self.weights_ = np.ones(self.n_components) / self.n_components
        P_mat = np.zeros((n_samples, self.n_components))  # 概率矩阵

        for i in range(self.max_iter):
            # 分别对K各类概率
            for k in range(self.n_components):
                self.covariances_ += self.reg_covar  # 防止出现奇异协方差矩阵
                g = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
                #### E-step，计算概率 ####
                P_mat[:, k] = self.weights_[k] * g.pdf(X_train)  # 计算X在各分布下出现的频率
                #print(self.weights_[k].shape)
                #print(X_train.shape)
                #print(g.pdf(X_train).shape)

            totol_N=P_mat.sum(axis=1)#按行求和，样本X1在所有类中出现的概率
            totol_N[totol_N==0]=self.n_components
            P_mat/=totol_N.reshape(-1,1)#x/y计算对应元素相除（矩阵点除）
            #print(self.means_[k])
            #print(self.covariances_[k])

            # M-step
            for k in range(self.n_components):
                N_k = np.sum(P_mat[:, k], axis=0)  # 类出现的频率
                self.means_[k] = (1 / N_k) * np.sum(X_train *
                                                    P_mat[:, k].reshape(-1, 1), axis=0)  # 该类的新均值
                self.covariances_[k] = (1 / N_k) * np.dot((P_mat[:, k].reshape(-1, 1)
                                                           * (X_train - self.means_[k])).T,
                                                          (X_train - self.means_[k])) + self.reg_covar
                self.weights_[k] = N_k / n_samples

    def predict(self,X_test):
        P_mat = np.zeros((X_test.shape[0], self.n_components))
        for k in range(self.n_components):
            g = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            P_mat[:, k] = self.weights_[k] * g.pdf(X_test)

        totol_N = P_mat.sum(axis=1)
        totol_N[totol_N == 0] = self.n_components
        P_mat /= totol_N.reshape(-1, 1)
        #### E-step，计算概率 ####
        return np.argmax(P_mat, axis=1)

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X, _ = make_blobs(cluster_std=1.5, random_state=42, n_samples=1000, centers=3)
    X = np.dot(X, np.random.RandomState(0).randn(2, 2))  # 生成斜形类簇
    lines=np.where(np.isinf(X))
    #print(lines)

    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
    #plt.show()

    X_train, X_test = train_test_split(X, test_size=0.2)
    n_samples, n_feature = X_train.shape
    print(X_train)

    print("============================================================================================================")

    gmm0= GaussianMixture0(n_components=6)
    gmm0.fit(X_train)
    Y_pred = gmm0.predict(X_test)
    print(Y_pred)

    plt.clf()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred, alpha=0.3)
    plt.show()








