import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import itertools
import json
import random
import time
from ast import literal_eval as make_tuple
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import psutil

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF, LatentDirichletAllocation

import os
print(os.listdir()) # check docker container

from configs import BASELINE_PATH, LOGGER, JSON_LOGGER
from utils.helper import now_int

class DenseTransformer(BaseEstimator):
    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None, **fit_params):
        from scipy.sparse import issparse
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X
    
    def fit(self, X, y=None, **fit_params):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X=X, y=y)

    

class PredictJob:
    def __init__(self, processor_name, processor_par, clf_name, clf_par, topic_name, topic_par, num_repeat: int = 1):
        self.topic_name = topic_name
        self.topic_par = topic_par
        self.processor_name = processor_name
        self.processor_par = processor_par
        self.clf_name = clf_name
        self.clf_par = clf_par
        self.result = None
        self.start_time = None
        self.done_time = None
        self.num_repeat = num_repeat

class JobWorker(Process):
    def __init__(self, pending_q: Queue) -> None:
        super().__init__()
        self.pending_q = pending_q
        X, self.Y = mnist_reader.load_mnist(path=DATA_DIR, kind='train')
        Xt, self.Yt = mnist_reader.load_mnist(path=DATA_DIR, kind='t10k')
        scaler = preprocessing.StandardScaler().fit(X)
        self.X = scaler.transform(X)
        self.Xt = scaler.transform(Xt)

    def run(self) -> None:
        while True:
            cur_job = self.pending_q.get()  # type: PredictJob

            LOGGER.info('job received! repeat: %d classifier: "%s" parameter: "%s" processor: "%s" parameter: "%s" topicmodel: "%s" parameter: "%s"' % (cur_job.num_repeat,
                                                                                       cur_job.clf_name,
                                                                                       cur_job.clf_par,
                                                                                       cur_job.processor_name,
                                                                                       cur_job.processor_par,
                                                                                       cur_job.topic_name,
                                                                                       cur_job.topic_par))
            if cur_job.clf_name in globals():
                try:
                    acc = []
                    cur_job.start_time = now_int()
                    for j in range(cur_job.num_repeat):
                        cur_score = self.get_accuracy(cur_job.processor_name, 
                                                      cur_job.processor_par,
                                                      cur_job.clf_name, 
                                                      cur_job.clf_par,
                                                      cur_job.topic_name,
                                                      cur_job.topic_par,
                                                      j)
                        acc.append(cur_score)
                        if len(acc) == 2 and abs(acc[0] - cur_score) < 1e-3:
                            LOGGER.info('%s is invariant to training data shuffling, will stop repeating!' %
                                        cur_job.clf_name)
                            break
                    cur_job.done_time = now_int()
                    test_info = {
                        'name': cur_job.clf_name,
                        'parameter': cur_job.clf_par,
                        'processor': cur_job.processor_name,
                        'processor_para': cur_job.processor_par,
                        'topic_model': cur_job.topic_name,
                        'topic_para': cur_job.topic_par,
                        'score': acc,
                        'start_time': cur_job.start_time,
                        'done_time': cur_job.done_time,
                        'num_repeat': len(acc),
                        'mean_accuracy': np.array(acc).mean(),
                        'std_accuracy': np.array(acc).std() * 2,
                        'time_per_repeat': int((cur_job.done_time - cur_job.start_time) / len(acc))
                    }

                    JSON_LOGGER.info(json.dumps(test_info, sort_keys=True))

                    LOGGER.info('done! acc: %0.3f (+/- %0.3f) repeated: %d classifier: "%s" '
                                'parameter: "%s" processor: "%s" processor_para: "%s" '
                                'topic_model: "%s" topicmodel_para: "%s" '% (np.array(acc).mean(),
                                                      np.array(acc).std() * 2,
                                                      len(acc),
                                                      cur_job.clf_name,
                                                      cur_job.clf_par,
                                                      cur_job.processor_name,
                                                      cur_job.processor_par,
                                                      cur_job.topic_name,
                                                      cur_job.topic_par))
                except Exception as e:
                    LOGGER.error('%s with %s failed! reason: %s' % (cur_job.clf_name, cur_job.clf_par, e))
            else:
                LOGGER.error('Can not found "%s" in scikit-learn, missing import?' % cur_job.clf_name)

    def get_accuracy(self, processor_name, processor_par, clf_name, clf_par, topic_name, topic_par, id):
        start_time = time.clock()
#        clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
        Xs, Ys = shuffle(self.X, self.Y)
        clf = make_pipeline(globals()[processor_name](**processor_par), 
                            DenseTransformer(), 
                            globals()[topic_name](**topic_par),
                            globals()[clf_name](**clf_par))
        scores = cross_val_score(
        clf, Xs, Ys, cv=5, scoring='f1_macro')
        cur_score = scores.mean()
#        cur_score = clf.fit(Xs, Ys).score(self.Xt, self.Yt)
        duration = time.clock() - start_time
        LOGGER.info('#test: %d acc: %0.3f time: %.3fs classifier: "%s" parameter: "%s" processor: "%s" processor_parameter: "%s"' % (id, cur_score,
                                                                                           duration,
                                                                                           clf_name,
                                                                                           clf_par,
                                                                                           processor_name,
                                                                                           processor_par))
        return cur_score


class JobManager:
    def __init__(self, num_worker: int = 2, num_repeat: int = 2, do_shuffle: bool = False,
                 respawn_memory_pct: float = 90):
        self.pending_q = Queue()
        self.num_worker = num_worker
        self.num_repeat = num_repeat
        self.do_shuffle = do_shuffle
        self.valid_jobs = self._sanity_check(self._parse_tasks(BASELINE_PATH))
        self.respawn_memory_pct = respawn_memory_pct
        for v in self.valid_jobs:
            self.pending_q.put(v)

    def memory_guard(self):
        LOGGER.info('memory usage: %.1f%%, RESPAWN_LIMIT: %.1f%%',
                    psutil.virtual_memory()[2], self.respawn_memory_pct)
        if psutil.virtual_memory()[2] > self.respawn_memory_pct:
            LOGGER.warn('releasing memory now! kill iterator processes and restart!')
            self.restart()

    def restart(self):
        self.close()
        self.start()

    def _parse_list(self, v):
        for idx, vv in enumerate(v):
            if isinstance(vv, str) and vv.startswith('('):
                v[idx] = make_tuple(vv)
        return v

    def _parse_tasks(self, fn):
        with open(fn) as fp:
            tmp = json.load(fp)

        def get_par_comb(tmp, clf_type, clf_name):
            all_par_vals = list(itertools.product(*[self._parse_list(vv)
                                                    for v in tmp[clf_type][clf_name]
                                                    for vv in v.values()]))
            all_par_name = [vv for v in tmp[clf_type][clf_name] for vv in v.keys()]
            return [{all_par_name[idx]: vv for idx, vv in enumerate(v)} for v in all_par_vals]

        processor_result = [{v: vv} for v in tmp['processor'] for vv in get_par_comb(tmp, 'processor', v)]
        clf_result = [{v: vv} for v in tmp['classifiers'] for vv in get_par_comb(tmp, 'classifiers', v)]
        topicmodels = [{v: vv} for v in tmp['topicmodels'] for vv in get_par_comb(tmp, 'topicmodels', v)]
        result = list(itertools.product(processor_result, clf_result, topicmodels))
        if self.do_shuffle:
            random.shuffle(result)
        return result

    def close(self):
        for w in self.workers:
            w.join(timeout=1)
            w.terminate()

    def start(self):
        self.workers = [JobWorker(self.pending_q) for _ in range(self.num_worker)]
        for w in self.workers:
            w.start()

    def _sanity_check(self, all_tasks):
        total_clf = 0
        failed_clf = 0
        Xt, Yt = mnist_reader.load_mnist(path=DATA_DIR, kind='t10k')
        Xt = preprocessing.StandardScaler().fit_transform(Xt)
        Xs, Ys = shuffle(Xt, Yt)
        num_dummy = 10
        Xs = Xs[:num_dummy]
        Ys = [j for j in range(10)]
        valid_jobs = []
        for v in all_tasks:
            processor_name = list(v[0].keys())[0]
            processor_par = list(v[0].values())[0]
            clf_name = list(v[1].keys())[0]
            clf_par = list(v[1].values())[0]
            topic_name = list(v[2].keys())[0]
            topic_par = list(v[2].values())[0]
            total_clf += 1
            try:
                make_pipeline(globals()[processor_name](**processor_par), 
                              DenseTransformer(), 
                              globals()[topic_name](**topic_par),
                              globals()[clf_name](**clf_par)).fit(Xs, Ys)
                valid_jobs.append(PredictJob(processor_name, processor_par, 
                                             clf_name, clf_par, 
                                             topic_name, topic_par, 
                                             self.num_repeat))
            except Exception as e:
                failed_clf += 1
                LOGGER.error('Can not create classifier "%s" with parameter "%s". Reason: %s' % (clf_name, clf_par, e))
        LOGGER.info('%d classifiers to test, %d fail to create!' % (total_clf, failed_clf))
        return valid_jobs


if __name__ == "__main__":
    # predicting()
    jm = JobManager()
    jm.start()
    # jm.start()