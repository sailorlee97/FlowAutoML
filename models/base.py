"""
@Time    : 2023/4/23 15:39
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: base.py
@Software: PyCharm
"""
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from IPy import IP
from sklearn.metrics import f1_score

class BaseClassification():
    '''Abstract class for all algorithms.

    '''
    def __init__(self,opt):
        self.opt = opt

    def _propress_label(self, labels,sorted_labels):
        """Encode target labels with value between 0 and n_classes-1.

        This transformer should be used to encode target values, *i.e.* `y`, and
        not the input `X`.

        :param data:
        :return: numic
        """
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)
        #
        # res = {}
        # for cl in le.classes_:
        #     res.update({cl: le.transform([cl])[0]})
        # print(res)
        reskey = {}
        for i in sorted_labels:
            reskey.update({i: sorted_labels.index(i)})
        print(reskey)
        # map映射
        labels = labels.map(reskey).values

        print(labels)
        return labels, reskey

    def _process_num(self, num):
        """
            This estimator scales and translates each feature individually such
        that it is in the given range on the training set, e.g. between
        zero and one.

        The transformation is given by::

            X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X_scaled = X_std * (max - min) + min

        where min, max = feature_range.

        This transformation is often used as an alternative to zero mean,
        unit variance scaling.
        :param num: dataframe or array
        :return:  array
        """
        mm = MinMaxScaler()
        X = mm.fit_transform(num)

        return X

    def _process_normalizer(self,x):
        """Normalize samples individually to unit norm.

        Each sample (i.e. each row of the data matrix) with at least one
        non zero component is rescaled independently of other samples so
        that its norm (l1, l2 or inf) equals one.

        This transformer is able to work both with dense numpy arrays and
        scipy.sparse matrix (use CSR format if you want to avoid the burden of
        a copy / conversion).

        Scaling inputs to unit norms is a common operation for text
        classification or clustering for instance. For instance the dot
        product of two l2-normalized TF-IDF vectors is the cosine similarity
        of the vectors and is the base similarity metric for the Vector
        Space Model commonly used by the Information Retrieval community.

        Read more in the :ref:`User Guide <preprocessing_normalization>`.

        :param x:
        :return:
        """

        scaler = Normalizer(norm='l2')
        scaler.fit(x)

        return scaler.fit_transform(x)

    def _process_standard(self,x):

        scaler = StandardScaler()
        scaler.fit(x)

        return scaler.transform(x)

    def ipToValue(self,ipseries):
        '''

        :param ipseries:
        :return:list
        '''
        ipd = []
        for i in ipseries:
            ip = IP(i)
            ipd.append(ip.int())
        return ipd

    def get_f1(self, y_test,y_pred):
        '''
        get f1
        :param y_test:
        :return:
        '''

        f1 = f1_score(y_test.argmax(-1), y_pred.argmax(-1), average='weighted')

        return f1

    def _writefeatures(self,path,featurelist):

        f = open(path, 'a')
        for i in featurelist:
            f.write(i)
            f.write('\n')
        f.close()