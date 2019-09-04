from flask import Flask, render_template, request, redirect, url_for
import tweepy #The Twitter API
from time import sleep
from datetime import datetime
from textblob import TextBlob #For Sentiment Analysis
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import re
import bs4
import requests
from wordcloud import WordCloud, STOPWORDS
from plotly.offline import plot
from plotly.graph_objs import Scatter
from plotly.graph_objs import Pie
from flask import Markup
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering


UPLOAD_FOLDER = '/static'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)

@app.route("/")
def hom():
	return render_template('index.html')

##############################               Twitter Analysis            ################################################
@app.route("/Tweets_Analysis")
def Tweets_Analysis():
		return render_template('Tweets_Analysis.html')

@app.route("/Tweets_Results",methods=['POST'])
def Tweets_Results():
	_keyword = request.form['topic']
	_number = request.form['tweets']
	_number = int(_number)
	consumer_key = '438zh7gYOYLFd5GE9Wt2HZDgz'
	consumer_secret = 'QjmMOToC7XXi2VxrUbEyvJrVqMeBTJj79Ef6IXMxVDodbh7DtQ'
	access_token = '2417664006-FuvsiGAO3KFZvuO7tHX2JKhxwgPiLuVgKLj0tlb'
	access_token_secret = 'iS67VqAxKYK3TOLojyEAN60fN9Bf5lq6qxJQbVegG5Qda'
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)
	user = api.me()
	print(user.name)
	results = api.search(
		lang="en",
		q=_keyword + " -rt",
		count=_number,
		result_type="recent"
		)
	tweets = []
	for i in results:
		tweets.append(i.text)
	polarity_list = []
	numbers_list = []
	number = 1
	for i in tweets:
		analysis = TextBlob(i)
		fanalysis = analysis.sentiment
		polarity = fanalysis.polarity
		polarity_list.append(polarity)
		numbers_list.append(number)
		number = number + 1

	averagePolarity = (sum(polarity_list))/(len(polarity_list))
	averagePolarity = "{0:.0f}%".format(averagePolarity * 100)
	time  = datetime.now().strftime("At: %H:%M\nOn: %m-%d-%y")

	#plotly
	my_plot_div = plot([Scatter(x=numbers_list, y=polarity_list,mode = "markers",text = tweets, marker = dict(size = 10))], output_type='div')
	print(_keyword,_number,averagePolarity)

	comment_words = ' '
	stopwords = set(STOPWORDS)
	stopwords.add("https")
	stopwords.add(_keyword)
	stopwords.add("co")
	for val in tweets:
	    val = str(val)

	        # split the value
	    tokens = val.split()

	        # Converts each token into lowercase
	    for i in range(len(tokens)):
	        tokens[i] = tokens[i].lower()

	    for words in tokens:
	        comment_words = comment_words + words + ' '

	mask = np.array(Image.open(requests.get('https://c7.uihere.com/files/354/747/724/logo-united-states-presidential-election-debates-2016-icon-twitter-png-image-thumb.jpg', stream=True).raw))

	wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10, mask=mask).generate(comment_words)

	plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.tight_layout(pad=0)

	plt.savefig("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/total_wordcloud.png")

	df = pd.DataFrame({"tweets":tweets,"polarity":polarity_list})
	a = df.loc[df["polarity"]<0]
	comment_words = ' '
	stopwords = set(STOPWORDS)
	stopwords.add("https")
	stopwords.add(_keyword)
	stopwords.add("co")

	for val in a["tweets"]:
	   val = str(val)
	   # split the value
	   tokens = val.split()

	        # Converts each token into lowercase
	   for i in range(len(tokens)):
	      tokens[i] = tokens[i].lower()

	   for words in tokens:
	      comment_words = comment_words + words + ' '

	mask = np.array(Image.open(requests.get('https://c7.uihere.com/files/354/747/724/logo-united-states-presidential-election-debates-2016-icon-twitter-png-image-thumb.jpg', stream=True).raw))

	wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10, mask=mask).generate(comment_words)

	plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.tight_layout(pad=0)

	plt.savefig("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/negative_cloud.png")

	tfidf_vectorizer = TfidfVectorizer(min_df=1,stop_words="english")

	simi = []
	b = list(a["tweets"])
	p = list(a["polarity"])
	n = []
	for i in range(0,len(b)):
		n.append(i)

	for i in range(0,len(b)):

		subsimi = []
		for j in range(0,len(b)):

			train_set = [b[i],b[j]]
			tfidf_mat = tfidf_vectorizer.fit_transform(train_set)
			a = cosine_similarity(tfidf_mat[0:1],tfidf_mat)
			subsimi.append(a[0][1])

		simi.append(subsimi)
	dis = np.array(simi)
	dis = np.round(dis,2)
	diss = 1- dis
	similarity = diss
	graph = similarity.copy()

	model = AgglomerativeClustering(affinity="precomputed", n_clusters = 3, linkage="complete").fit(graph)
	print(model.labels_)
	clust = model.labels_

	my_plot_div2 = plot([Scatter(x=n, y=p,mode = "markers",text = b, marker = dict(size = 10, color = clust,colorscale='Viridis',showscale=True))], output_type='div')
	return render_template("Tweets_Results.html", T = time, ASentiment = averagePolarity,total = _number, topic = _keyword, div_placeholder=Markup(my_plot_div),second_plot = Markup(my_plot_div2))

############################                File Analysis                  ##################################################
@app.route("/File_Analysis")
def File_Analysis():
	return render_template('File_Analysis.html')

@app.route("/File_Results", methods = ['GET', 'POST'])
def File_Results():
	if request.method == 'POST':
		f = request.files['file']
		f.filename = "senti.csv"
		f.save("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/"+f.filename)

	try:
		df = pd.read_csv("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/senti.csv")
		polarity_list = []
		numbers_list = []
		number = 1
		polar = []
		df1 = list(df.iloc[:,0])
		for i in df1:
			analysis = TextBlob(i)
			fanalysis = analysis.sentiment
			polarity = fanalysis.polarity
			polarity_list.append(polarity)
			numbers_list.append(number)
			number = number + 1

		negative = 0
		positive = 0
		neutral = 0
		for i in polarity_list:
			if i < 0:
				polar.append("Negative")
				negative += 1
			if i == 0:
				polar.append("Neutral")
				neutral += 1
			else:
				polar.append("Positive")
				positive += 1
		total_text = negative + positive + neutral
		averagePolarity = (sum(polarity_list))/(len(polarity_list))
		averagePolarity = "{0:.0f}%".format(averagePolarity * 100)
		time  = datetime.now().strftime("At: %H:%M\nOn: %m-%d-%y")
		values = [negative,neutral,positive]

		my_plot_pie = plot([Pie(values = values, labels = ["Negative","Neutral","Positive"])], output_type='div')

		print("all ok 4")
		comment_words = ' '
		stopwords = set(STOPWORDS)
		df = pd.DataFrame({"text":df1,"polarity":polarity_list})
		a = df.loc[df["polarity"]<0]
		comment_words = ' '
		stopwords = set(STOPWORDS)

		for val in a["text"]:
			val = str(val)

            # split the value
			tokens = val.split()

            # Converts each token into lowercase
			for i in range(len(tokens)):
				tokens[i] = tokens[i].lower()
			for words in tokens:
				comment_words = comment_words + words + ' '

		wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10).generate(comment_words)
		plt.figure(figsize = (5, 5), facecolor = None)
		plt.imshow(wordcloud)
		plt.axis("off")
		plt.tight_layout(pad = 0)
		plt.savefig("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/file_negative_cloud.png")
#######################################
		tfidf_vectorizer = TfidfVectorizer(min_df=1,stop_words="english")

		simi = []
		b = list(a["text"])
		p = list(a["polarity"])
		n = []
		for i in range(0,len(b)):
			n.append(i)

		for i in range(0,len(b)):

			subsimi = []
			for j in range(0,len(b)):

				train_set = [b[i],b[j]]
				tfidf_mat = tfidf_vectorizer.fit_transform(train_set)
				a = cosine_similarity(tfidf_mat[0:1],tfidf_mat)
				subsimi.append(a[0][1])

			simi.append(subsimi)
		dis = np.array(simi)
		dis = np.round(dis,2)
		diss = 1- dis
		similarity = diss
		graph = similarity.copy()

		model = AgglomerativeClustering(affinity="precomputed", n_clusters = 3, linkage="complete").fit(graph)
		print(model.labels_)
		clust = model.labels_

		my_plot_file = plot([Scatter(x=n, y=p,mode = "markers",text = b, marker = dict(size = 10, color = clust,colorscale='Viridis',showscale=True))], output_type='div')

		return render_template("File_Results.html", ASentiment = averagePolarity,total = total_text, negative = negative, positive = positive, neutral = neutral, pie = Markup(my_plot_pie),file = Markup(my_plot_file))
	except:
		return 'please upload a csv file'

##########################               Course Analysis               ###################################################
@app.route("/Course_Analysis", methods = ['GET', 'POST'])
def Course_Analysis():
	if request.method == 'POST':
		_courseName = request.form['course']
		df = pd.read_excel('C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/output.xlsx')
		df.head()
		review_lst = []
		s_polarity = []
		positive_reviews_lst = []
		negative_reviews_lst = []
		positive_reviews = 0
		negative_reviews = 0
		neutral_reviews = 0
		course_input = _courseName
		course_input = course_input.lower()
		df= df.applymap(lambda s:s.lower() if type(s) == str else s)
		df = df.set_index("Course_name", drop = False)
		try:
			if df['Course_name'].str.contains(str(course_input)).any():
				#print("Course is available")
				is_course_in_name = df['Course_name'].str.contains(str(course_input))
				course_in_name = df[is_course_in_name]
				#print(course_in_name.head())
				course_lst = list((course_in_name.loc[:,"Course_name"]))
				#print(course_lst)
				course_lst_no = len(course_lst)
			else:
				course_lst = ['No Such Courses available']
			return render_template("Course_Analysis.html", Course_header = 'Available Courses', course_lst_no = "["+str(course_lst_no)+"]", course_lst = course_lst)
		except:
			return render_template("Course_Analysis.html", Course_header = 'No Courses Available')
	else:
		return render_template('Course_Analysis.html')

@app.route("/Course_Results", methods = ['GET', 'POST'])
def Course_Results():
	_courseName = request.form['course']
	df = pd.read_excel('C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/output.xlsx')
	df.head()
	review_lst = []
	s_polarity = []
	n_polarity = []
	positive_reviews_lst = []
	negative_reviews_lst = []
	positive_reviews = 0
	negative_reviews = 0
	neutral_reviews = 0
	course_input = _courseName
	course_input = course_input.lower()
	df= df.applymap(lambda s:s.lower() if type(s) == str else s)
	df = df.set_index("Course_name", drop = False)
	try:
		if df['Course_name'].str.contains(str(course_input)).any():
		    print("Course is available")
		    print(course_input)
		    is_course_in_name = df['Course_name'].str.contains(str(course_input))
		    course_in_name = df[is_course_in_name]
		    #print(course_in_name.head())
		    course_lst = list((course_in_name.loc[:,"Course_name"]))
		    print(course_lst)
		    url = (course_in_name.loc[str(course_input),"course_url"])
		    review_no = (course_in_name.loc[str(course_input),"review_nos"])
		    print(url)
		    print(review_no)
		    if review_no == 0:
		    	print("No Reviews Available")
		    	return render_template("Course_Analysis.html", Course_header = 'No Reviews Available')
		    else:
		    	try:
		    		page = requests.get(url)
		    		soup = bs4.BeautifulSoup(page.text,'html.parser')
		    		result = soup.findAll("div",{"class":"review-body__content"})
		    	except:
		    		return "Please Check Internet Connection"

		    	for res in result:
		    		review = res.find("span",{"class":"more-less-trigger__text--full"}).text.strip()
		    		review_lst.append(review)
		    		#print(review_lst)
		    		#print(len(review_lst))
		    	for page_no in range(2,10):
		    		if len(review_lst) == int(review_no):
		    			break
		    		p_url = str(url)+"?page="+str(page_no)+"#incourse-reviews"
		    		page = requests.get(p_url)
		    		soup = bs4.BeautifulSoup(page.text,'html.parser')
		    		result = soup.findAll("div",{"class":"review-body__content"})
		    		for res in result:
		    			review = res.find("span",{"class":"more-less-trigger__text--full"}).text.strip()
		    			review_lst.append(review)

		    	print(len(review_lst))
		    	#print(review_lst)
		    	for r in range(0,len(review_lst)):
		    		analysis = TextBlob(review_lst[r])
		    		s_polarity.append(analysis.sentiment.polarity)
		    		#if analysis.detect_language() == 'en':
		    		#print(analysis.sentiment)
		    		#print(analysis.sentiment.polarity)
		    		if analysis.sentiment[0]>0:
		    			positive_reviews = positive_reviews + 1
		    			positive_reviews_lst.append(review_lst[r])
		    		elif analysis.sentiment[0]<0:
		    			negative_reviews = negative_reviews + 1
		    			negative_reviews_lst.append(review_lst[r])
		    			n_polarity.append(analysis.sentiment.polarity)
		    		else:
		    			neutral_reviews = neutral_reviews + 1
		    		total_reviews = positive_reviews + negative_reviews + neutral_reviews
		else:
			print("Course is not available")
		def Average(lst):
			return sum(lst) / len(lst)
		sentiment_polarity = round(Average(s_polarity),2)
		print("Total Reviews = " + str(total_reviews))
		print("Positive Reviews = " + str(positive_reviews))
		print("Negative Reviews = " + str(negative_reviews))
		print("Neutral Reviews = " + str(neutral_reviews))
		print("Sentiment_Polarity = " + str(round(sentiment_polarity,2)))

		# Scatter plot
		plt.figure(figsize = (4.5, 4.5), facecolor = None)
		x = range(0, total_reviews)
		y = s_polarity
		colors = ("blue")
		plt.scatter(x, y, c=colors, alpha=0.8)
		plt.title('Scatter plot of Course Reviews')
		plt.xlabel('No of Reviews')
		plt.ylabel('Sentiment Polarity')
		plt.axhline(0, color='green')
		plt.savefig("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/course_scatter.png")

		# Pie Chart
		#plt.figure(figsize = (5, 5), facecolor = None)
		labels = ['positive', 'negative', 'neutral']
		sizes = [positive_reviews, negative_reviews, neutral_reviews]
		#colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
		#explode = (0, 0, 0)
		#plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
		#plt.axis('equal')

		course_plot_pie = plot([Pie(values = sizes, labels = labels)], output_type='div')

		# World Cloud Positive
		comment_words = ' '
		stopwords = set(STOPWORDS)
		for val in positive_reviews_lst:
			val = str(val)
			# split the value
			tokens = val.split()
			# Converts each token into lowercase
			for i in range(len(tokens)):
				tokens[i] = tokens[i].lower()
			for words in tokens:
				comment_words = comment_words + words + ' '
		wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10).generate(comment_words)
		plt.figure(figsize = (5, 5), facecolor = None)
		plt.imshow(wordcloud)
		plt.axis("off")
		plt.tight_layout(pad = 0)
		plt.savefig("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/course_positive_cloud.png")

		# World Cloud Negative
		if len(negative_reviews_lst) != 0:
			comment_words = ' '
			stopwords = set(STOPWORDS)
			for val in negative_reviews_lst:
				val = str(val)
				# split the value
				tokens = val.split()
				# Converts each token into lowercase
				for i in range(len(tokens)):
					tokens[i] = tokens[i].lower()
				for words in tokens:
					comment_words = comment_words + words + ' '
			wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10).generate(comment_words)
			plt.figure(figsize = (5, 5), facecolor = None)
			plt.imshow(wordcloud)
			plt.axis("off")
			plt.tight_layout(pad = 0)
			plt.savefig("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/course_negative_cloud.png")

		else:
			negative_reviews_lst = ['Zero Negative Reviews']
			comment_words = ' '
			stopwords = set(STOPWORDS)
			for val in negative_reviews_lst:
				val = str(val)
				# split the value
				tokens = val.split()
				# Converts each token into lowercase
				for i in range(len(tokens)):
					tokens[i] = tokens[i].lower()
				for words in tokens:
					comment_words = comment_words + words + ' '
			wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10).generate(comment_words)
			plt.figure(figsize = (5, 5), facecolor = None)
			plt.imshow(wordcloud)
			plt.axis("off")
			plt.tight_layout(pad = 0)
			plt.savefig("C:/Users/Visheshwar/Desktop/Aegis School of Business/Python/Evaluation/flask_pro/static/course_negative_cloud.png")

		return render_template("Course_Results.html", ASentiment = sentiment_polarity,total = total_reviews, negative = negative_reviews, positive = positive_reviews, neutral = neutral_reviews, pie = Markup(course_plot_pie), Course_Name = course_input.title())
	except:
		return "Please type the exact name of the course"

@app.route("/Details")
def Project_Details():
	return render_template('Project_Details.html')

@app.route("/About")
def About_Us():
	return render_template('About_Us.html')


if __name__ == '__main__':
	app.run(debug=True)
