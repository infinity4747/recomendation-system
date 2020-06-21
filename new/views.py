import numpy as np
import pandas as pd
import os
import csv
import sys
import re
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import pickle as p
from django.shortcuts import render
from surprise import SVD, Dataset
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from rest_framework import pagination

ratingsPath = "/Users/infinity/Downloads/ratings.csv"
moviesPath = "/Users/infinity/Downloads/movies.csv"
movieID_to_name = {}
def index(request):
	df = pd.read_csv(moviesPath)
	page = request.GET.get('page', 1)
	paginator = Paginator(df.title[:100],15)
	return render(request,'index.html',{"data":df.title[:40]})

def loadMovieLensLatestSmall():

        # Look for files relative to the directory we are running from
        # os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        name_to_movieID = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(ratingsPath, reader=reader)

        with open(moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0])
                    movieName = row[1]
                    movieID_to_name[movieID] = movieName

        return ratingsDataset

def getMovieName(movieID):
    if movieID in movieID_to_name:
        return movieID_to_name[movieID]
    else:
        return ""

def BuildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []

    u = trainset.to_inner_uid(str(testSubject))

    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset

def result(id):
    a=id
   
    a = a.replace("2C", "")
    i=0
    list1=a.split("%")
    print(list1)
    return list1


movieids = []
def makecalc(request):
    list_id=request.GET['res']
    list_id = list_id.replace("2C", "")
    new_list_id=list_id.split("%")[0]
    movieIds = new_list_id.split(",")
    for id in movieIds:
        movieids.append(id)
    print("Here\n\n")
    print(movieids)
    df = pd.read_csv(ratingsPath)

    uid = df['userId'].iloc[-1]+1
    for i in range (0,len(movieids)):
        df2=pd.Series([uid, movieids[i], 5.0,df.timestamp.mean()], index=df.columns )
        df=df.append(df2,ignore_index=True)

    df.to_csv(ratingsPath, index=False)
    data = loadMovieLensLatestSmall()
    trainSet = data.build_full_trainset()
    testSet = BuildAntiTestSetForUser(uid, trainSet)
    algo = SVD()
    algo.fit(trainSet)
    # print(testSet)
    predictions = algo.test(testSet)

    recommendations = []

    print ("\nWe recommend:")
    for userID, movieID, actualRating, estimatedRating, _ in predictions:
        try:
            intMovieID = int(float(movieID))
            recommendations.append((intMovieID, estimatedRating))
        except Exception as e:
        	pass
        
        

    recommendations.sort(key=lambda x: x[1], reverse=True)
    ans = []
    for ratings in recommendations[:10]:
        print(getMovieName(ratings[0]))
        ans.append(getMovieName(ratings[0]))
    page = request.GET.get('page', 1)
    paginator = Paginator(ans,15)
    try:
        data = paginator.page(page)
    except PageNotAnInteger:
        data = paginator.page(1)
    except EmptyPage:
        data = paginator.page(paginator.num_pages)
    return render(request,'properties.html',{"data":data})
