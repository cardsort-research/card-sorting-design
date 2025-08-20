import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import copy, random, os, json
from sklearn.metrics import cohen_kappa_score
from skbio.stats.distance import mantel
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, f_oneway, ttest_ind
from scipy.stats import chi2_contingency, shapiro, spearmanr, linregress
from skbio.stats.distance import mantel
import math, sys, re
import itertools, seaborn as sns
from IPython.display import HTML
from wordcloud import WordCloud
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from kneed import KneeLocator
from wordfreq import word_frequency
from functools import reduce
from collections import Counter
from scipy.stats import kruskal
from scipy.optimize import curve_fit
import warnings
sns.set_theme()

np.seterr(divide='ignore', invalid='ignore')

def informativeness(text):
    words = re.findall(r"\b\w+'\w+|\w+\b", text.lower())
    totalSurprisal = 0
    for word in words:
        frequency = word_frequency(word, 'en', wordlist='large', minimum=0.0)
        surprisal = -math.log2(frequency) if frequency != 0 else 0
        totalSurprisal += surprisal
    return totalSurprisal

def loadData():
    respondents = pd.read_csv(os.path.join('..', 'data', 'respondents.csv'), index_col=0)
    results = pd.read_csv(os.path.join('..', 'data', 'results.csv'), index_col=0)
    cardsE = pd.read_csv(os.path.join('..', 'data', 'cards', 'cardsE.csv'), index_col=0)
    cardsB = pd.read_csv(os.path.join('..', 'data', 'cards', 'cardsB.csv'), index_col=0)
    return respondents, results, cardsE.index.values, cardsB.index.values

def BMM(inputMatrix):
    simMatrix = np.copy(inputMatrix)
    clusters = [[x] for x in range(len(simMatrix))]
    #clusterHistory = [copy.deepcopy(clusters)]
    clusterHistory = []
    while(len(clusters) != 1):

        np.fill_diagonal(simMatrix, -1)
        first = np.argmax(simMatrix) // len(simMatrix)
        second = np.argmax(simMatrix) % len(simMatrix)
        np.fill_diagonal(simMatrix, 0)
        first, second = (second, first) if first > second else (first, second)

        fir = clusters[first]
        sec = clusters[second]
        agg = simMatrix[first, second]

        clusters[first] = clusters[first] + clusters[second]
        clusters.pop(second)
        
        #simMatrix[first, :] = (simMatrix[first, :] + simMatrix[second, :]) / 2
        simMatrix[first, :] = np.max([simMatrix[first, :], simMatrix[second, :]], axis=0)
        #print(simMatrix[first, :])
        np.fill_diagonal(simMatrix, 0)
        simMatrix = np.delete(simMatrix, (second), axis=0)
        #simMatrix[:, first] = (simMatrix[:, first] + simMatrix[:, second]) / 2
        simMatrix[:, first] = np.max([simMatrix[:, first], simMatrix[:, second]], axis=0)
        simMatrix = np.delete(simMatrix, (second), axis=1)
        np.fill_diagonal(simMatrix, 0)

        clusterHistory.append({'clusters': copy.deepcopy(clusters), 'agreement': float(agg), 'first': fir, 'second': sec })
    return clusters, clusterHistory


def AAM(inputMatrix):
    simMatrix = np.copy(inputMatrix)
    clusters = [[x] for x in range(len(simMatrix))]
    clusterHistory = []
    np.fill_diagonal(simMatrix, -1)
    
    while(not np.all(simMatrix == -1) and len(clusters) != 1):
        
        first = np.argmax(simMatrix) // len(simMatrix)
        second = np.argmax(simMatrix) % len(simMatrix)
        first, second = (second, first) if first > second else (first, second)

        agg = simMatrix[first, second]

        simMatrix[first, second] = -1
        simMatrix[second, first] = -1

        for i, c in enumerate(clusters):
            if(first in c):
                first = i
                break
        
        for i, c in enumerate(clusters):
            if(second in c):
                second = i
                break

        if(first == second):
            continue
        
        fir=clusters[first]
        sec=clusters[second]
        
        clusters[first] = clusters[first] + clusters[second]
        clusters.pop(second)
        
        clusterHistory.append({'clusters': copy.deepcopy(clusters), 'agreement': float(agg), 'first': fir, 'second': sec})

    return clusters, clusterHistory

def makeMatrix(type, data, labels, participants, aam=False, clusterHistory=False, clusteredOrder=[]):
    labelsToIndex = {x[1]: x[0] for x in enumerate(labels)}
    indexToLabels = {x[0]: x[1] for x in enumerate(labels)}

    matrix = {}
    for m in ['paired', 'seen', 'similarity', 'similarityAbsolute', 'C-similarity', 'C-similarityAbsolute']:
        matrix[m] = np.matrix([[0 for x in range(50)] for y in range(50)])

    for index, group in data.groupby(['respondent', 'categoryId']):
        cards = group.card.values
        pairs = [(a, b) for idx, a in enumerate(cards) for b in cards[idx + 1:]]
        pairs = list(map(lambda x: (labelsToIndex[x[0]], labelsToIndex[x[1]]), pairs))
        for pair in pairs:
            matrix['paired'][pair[1], pair[0]] += 1
            matrix['paired'][pair[0], pair[1]] += 1
    
    for index, group in data.groupby(['respondent']):
        cards = group.card.values
        pairs = [(a, b) for idx, a in enumerate(cards) for b in cards[idx + 1:]]
        pairs = list(map(lambda x: (labelsToIndex[x[0]], labelsToIndex[x[1]]), pairs))
        for pair in pairs:
            matrix['seen'][pair[1], pair[0]] += 1
            matrix['seen'][pair[0], pair[1]] += 1
    
    matrix['similarity'] = np.nan_to_num(matrix['paired'] / matrix['seen'] * 100)
    matrix['similarityAbsolute'] = np.nan_to_num(matrix['paired'] / participants * 100)
    matrix['C-similarity'] = np.nan_to_num(matrix['paired'] / matrix['seen'] * 100)
    matrix['C-similarityAbsolute'] = np.nan_to_num(matrix['paired'] / participants * 100)

    if(type.startswith('C-')):

        if(not aam):
            orderS, histS = BMM(matrix['similarity'])
            orderSA, histSA = BMM(matrix['similarityAbsolute'])
        else:
            orderS, histS = AAM(matrix['similarity'])
            orderSA, histSA = AAM(matrix['similarityAbsolute'])

        if(len(clusteredOrder)):
            orderS[0] = list(map(lambda x: labelsToIndex[x], clusteredOrder))
            orderSA[0] = list(map(lambda x: labelsToIndex[x], clusteredOrder))

        labelsNewS = list(map(lambda x: indexToLabels[x], orderS[0]))
        labelsNewSA = list(map(lambda x: indexToLabels[x], orderSA[0]))

        if(clusterHistory):
            historyS = []
            for index, i in enumerate(histS):
                historyS.append(i)
                for jndex, j in enumerate(historyS[-1]['clusters']):
                    historyS[-1]['clusters'][jndex] = list(map(lambda x: indexToLabels[x], j))
                historyS[-1]['first'] = list(map(lambda x: indexToLabels[x], historyS[-1]['first']))
                historyS[-1]['second'] = list(map(lambda x: indexToLabels[x], historyS[-1]['second']))
            
            historySA = []
            for index, i in enumerate(histSA):
                historySA.append(i)
                for jndex, j in enumerate(historySA[-1]['clusters']):
                    historySA[-1]['clusters'][jndex] = list(map(lambda x: indexToLabels[x], j))
        
        for i, ii in enumerate(orderS[0]):
            for j, jj in enumerate(orderS[0]):
                matrix['C-similarity'][i, j] = float(matrix['similarity'][ii, jj])

        matSA = np.matrix([[0.0 for x in range(50)] for y in range(50)])
        for i, ii in enumerate(orderSA[0]):
            for j, jj in enumerate(orderSA[0]):
                matrix['C-similarityAbsolute'][i, j] = float(matrix['similarityAbsolute'][ii, jj])

    dataF = pd.DataFrame(matrix[type])
    dataF.index = labels
    dataF.columns = labels

    if(type.startswith('C-')):
        if(type=='C-similarity'):
            dataF.index = labelsNewS
            dataF.columns = labelsNewS
        elif(type=='C-similarityAbsolute'):
            dataF.index = labelsNewSA
            dataF.columns = labelsNewSA

    if(type.startswith('C-') and clusterHistory):
        if(type.endswith('Absolute')):
            return dataF, historySA
        else:
            return dataF, historyS

    return dataF

def finalMatrices():
    _, results, cardsE, cardsB = loadData()
    matrices = {}
    for m in ['paired', 'seen', 'similarity', 'similarityAbsolute', 'C-similarity', 'C-similarityAbsolute']:
        matrices[m] = {}
        for v in ['E50', 'E30', 'B50', 'B30']:
            mat = makeMatrix(m, results[results.variant==v], cardsE if v in ['E50', 'E30'] else cardsB, 40)
            matrices[m][v] = {
                'data': mat,
                'cards': mat.index.values
            }
    return matrices

def plotVariants(respondents, attr, type, order=None, maxY=None):
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    print(respondents.groupby(['variant', attr]).size())
    if(type == 'bar'):
        if(order):
            respondents[respondents.variant=='E50'].groupby(attr).size().loc[[x for x in order if x in respondents[respondents.variant=='E50'].groupby(attr).size().index]].plot.bar(ax=ax1, title='E50 - ' + attr)
            respondents[respondents.variant=='E30'].groupby(attr).size().loc[[x for x in order if x in respondents[respondents.variant=='E30'].groupby(attr).size().index]].plot.bar(ax=ax2, title='E30 - ' + attr)
            respondents[respondents.variant=='B50'].groupby(attr).size().loc[[x for x in order if x in respondents[respondents.variant=='B50'].groupby(attr).size().index]].plot.bar(ax=ax3, title='B50 - ' + attr)
            respondents[respondents.variant=='B30'].groupby(attr).size().loc[[x for x in order if x in respondents[respondents.variant=='B30'].groupby(attr).size().index]].plot.bar(ax=ax4, title='B30 - ' + attr)
        else:
            respondents[respondents.variant=='E50'].groupby(attr).size().plot.bar(ax=ax1, title='E50 - ' + attr)
            respondents[respondents.variant=='E30'].groupby(attr).size().plot.bar(ax=ax2, title='E30 - ' + attr)
            respondents[respondents.variant=='B50'].groupby(attr).size().plot.bar(ax=ax3, title='B50 - ' + attr)
            respondents[respondents.variant=='B30'].groupby(attr).size().plot.bar(ax=ax4, title='B30 - ' + attr)
    if(type == 'hist'):
        respondents[respondents.variant=='E50'][attr].plot.hist(ax=ax1, title='E50 - ' + attr, )
        respondents[respondents.variant=='E30'][attr].plot.hist(ax=ax2, title='E30 - ' + attr)
        respondents[respondents.variant=='B50'][attr].plot.hist(ax=ax3, title='B50 - ' + attr)
        respondents[respondents.variant=='B30'][attr].plot.hist(ax=ax4, title='B30 - ' + attr)
        if(maxY):
            ax1.set_ylim(top=maxY)
            ax2.set_ylim(top=maxY)
            ax3.set_ylim(top=maxY)
            ax4.set_ylim(top=maxY)
    plt.tight_layout()

def plotMatrices(latestMatrices, attr):
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(14, 12))
    sns.heatmap(latestMatrices[attr]['E50']['data'], ax=ax1, 
                mask=np.triu(np.ones_like(latestMatrices[attr]['E50']['data'], dtype=bool)),
                vmin=0, vmax=100, cmap='binary', xticklabels=[], yticklabels=[])
    sns.heatmap(latestMatrices[attr]['E30']['data'], ax=ax2, 
                mask=np.triu(np.ones_like(latestMatrices[attr]['E30']['data'], dtype=bool)), 
                vmin=0, vmax=100, cmap='binary', xticklabels=[], yticklabels=[])
    sns.heatmap(latestMatrices[attr]['B50']['data'], ax=ax3, 
                mask=np.triu(np.ones_like(latestMatrices[attr]['B50']['data'], dtype=bool)), 
                vmin=0, vmax=100, cmap='binary', xticklabels=[], yticklabels=[])
    sns.heatmap(latestMatrices[attr]['B30']['data'], ax=ax4, 
                mask=np.triu(np.ones_like(latestMatrices[attr]['B30']['data'], dtype=bool)),
                vmin=0, vmax=100, cmap='binary', xticklabels=[], yticklabels=[])
    ax1.set_title('E50 - ' + attr)
    ax2.set_title('E30 - ' + attr)
    ax3.set_title('B50 - ' + attr)
    ax4.set_title('B30 - ' + attr)
    plt.tight_layout()

def my_mann(columns, g1, g2):
    res = []
    for i in columns:
        stat, p = mannwhitneyu(g1[i], g2[i])
        n1 = len(g1[i])
        n2 = len(g2[i])
        z = (stat - (n1*n2)/2)/math.sqrt((n1*n2*(n1+n2+1))/12)
        res.append({
            'feature':i, 'stat':stat, 'p':p, 'p<.001': 'yes' if p < .05 else '',
            'n1': n1, 'n2': n2, 'z': z,
            'effsize': z/math.sqrt(n1+n2),
            'm1': g1[i].mean(),
            'm2': g2[i].mean(),
            's1': g1[i].std(),
            's2': g2[i].std(),
            'med1': g1[i].median(),
            'med2': g2[i].median(),
            'q125': g1[i].quantile(.25),
            'q225': g2[i].quantile(.25),
            'q175': g1[i].quantile(.75),
            'q275': g2[i].quantile(.75)
        })
    return pd.DataFrame(res)

def getCategoryWords(indata):
    fixes = {
        'appliance': 'appliances',
        'applicances': 'appliances',
        'appliences': 'appliances',
        'camera': 'cameras',
        'computer': 'computers',
        'electonic': 'electronics',
        'electrical': 'electronics',
        'electricals': 'electronics',
        'electronic': 'electronics',
        'equipments': 'equipment',
        'gadget': 'gadgets',
        'miscallaneous': 'miscellaneous',
        'other': 'others',
        'part': 'parts',
        'phone': 'phones',
        'randoms': 'random',
        'tablet': 'tablets',
        'musical': 'music',
        'smaller': 'small',
        'account': 'accounts',
        'advise': 'advice',
        'app': 'apps',
        'calculator': 'calculators',
        'card': 'cards',
        'customer': 'customers',
        'claim': 'claims',
        'debt': 'debts',
        'finance': 'finances',
        'insuarance': 'insurance',
        'insurances': 'insurance',
        'invesments': 'investments',
        'investements': 'investments',
        'investment': 'investments',
        'loan': 'loans',
        'mortgage': 'mortgages',
        'other': 'others',
        'outgoing': 'outgoings',
        'payment': 'payments',
        'pension': 'pensions',
        'saving': 'savings',
        'service': 'services',
        'student': 'students',
        'trading': 'tranding',
        'caeds': 'cards',
        'educational': 'education',
        'everyd': 'everyday',
    }
    fixed = ' '.join(indata['categoryEdited'].values).split(' ')
    fixed = list(map(lambda x: fixes[x] if x in fixes else x , fixed))
    return fixed

def calculateWordOcurences(data1, data2):
    unique1, counts1 = np.unique(data1, return_counts=True)
    unique2, counts2 = np.unique(data2, return_counts=True)
    all = list(set(unique1.tolist() + unique2.tolist()))
    data = pd.DataFrame(index=all, columns=['first', 'second'])
    data['first'] = {k: v for k, v in sorted(dict(zip(unique1.tolist(), counts1.tolist())).items(), key=lambda item: item[1], reverse=True)}
    data['second'] = {k: v for k, v in sorted(dict(zip(unique2.tolist(), counts2.tolist())).items(), key=lambda item: item[1], reverse=True)}
    data['ratio'] = data['first']/data.second
    data['firstRel'] = data['first'] / len(data1) * 100
    data['secondRel'] = data.second / len(data2) * 100
    data['ratioRel1'] = data.firstRel/data.secondRel
    data['ratioRel2'] = data.secondRel/data.firstRel
    data['max'] = data[['firstRel', 'secondRel']].max(axis=1)
    for index, row in data.iterrows():
        pow, p, d, _ = chi2_contingency([[row['first'], len(data1)], [row['second'], len(data2)]])
        data.loc[index, 'chi'] = pow
        data.loc[index, 'p'] = p
        data.loc[index, 'd'] = d
        data.loc[index, 'n'] = len(data1) + len(data2)
        data.loc[index, '<p'] = 'yes' if p <=0.05 else ''
    return data

def getHalves(respondents, results, variant, feature):
    i1 = respondents[respondents.variant==variant].sort_values(feature)[:20].respondent.values
    i2 = respondents[respondents.variant==variant].sort_values(feature)[20:].respondent.values
    r1 = results[(results.respondent.isin(i1)) & (results.variant==variant)]
    r2 = results[(results.respondent.isin(i2)) & (results.variant==variant)]
    return r1, r2

def getClustersKmeans(data, max):
    m = 100 - data
    mds = MDS(random_state=2, dissimilarity='precomputed', n_components=3)
    embedding = mds.fit_transform(m)
    wcss = []
    clusters = []
    for i in range(2, max - 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        clusters.append(kmeans.fit_predict(embedding))
        wcss.append(kmeans.inertia_)
    knee = KneeLocator(range(2, max - 1), wcss, curve='convex', direction='decreasing')
    return (knee.elbow, clusters[knee.elbow - 2])