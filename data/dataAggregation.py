import csv
import glob
import datetime 
import json

def getETF(date):
	"""Takes in a date (str) and outputs corresponding data array for all ETFs"""

	path = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/ETFs/*.txt"
	files = glob.glob(path)
	stockData = []
	for file in files:
		with open(file) as f:
			csv_reader = csv.reader(f, delimiter = ",")
			for row in csv_reader:
				if row[0] == date:
					stockData.append(row)
	#print(stockData[0:2])
	return stockData

def getStock(date):
	"""Takes in a date (str) and outputs corresponding data array for all Stocks"""

	path = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/Stocks/*.txt"
	files = glob.glob(path)
	stockData = []
	for file in files:
		with open(file) as f:
			csv_reader = csv.reader(f, delimiter = ",")
			for row in csv_reader:
				if row[0] == date:
					stockData.append(row)

	return stockData


def getETF1(date, name):
	"""Takes in a date (str) and outputs corresponding data for specified ETF name (str) on that date"""
	name = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/ETFs/" + name + ".us.txt"
	path = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/ETFs/*.txt"
	files = glob.glob(path)
	print(files[0])
	stockData = []
	
	for file in files:
		if etf == file:
			with open(file) as f:
				csv_reader = csv.reader(f, delimiter = ",")
				for row in csv_reader:
					if row[0] == date:
						stockData.append(row)
	#print(stockData)
	return stockData

					

"""
def getETF5(initialDate, name):
	"Takes in an initial date (str) and ETF name (str), outputs data array consisting of entries up to 5 days prior to initial date"

	name = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/ETFs/" + name + ".us.txt"
	path = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/ETFs/*.txt"
	files = glob.glob(path)
	stockData = []

	date = datetime.date(*map(int, initialDate.split('-')))

	prevday1 = str(date - datetime.timedelta(days =1))
	prevday2 = str(date - datetime.timedelta(days =2))
	prevday3 = str(date - datetime.timedelta(days =3))
	prevday4 = str(date - datetime.timedelta(days =4))
	prevday5 = str(date - datetime.timedelta(days =5))

	dates = [initialDate, prevday1, prevday2, prevday3, prevday4, prevday5]

	for file in files:
		if name == file:
			with open(file) as f:
				csv_reader = csv.reader(f, delimiter = ",")
				for row in csv_reader:
					if row[0] in dates:
						stockData.append(row)
	#print(stockData)
	return stockData
"""


def getETF5(initialDate, name, category, days):
	""" 
	Input: initial date (str), ETF name (str), data category (int), days (int), 
	Data categories: Open: 1, High: 2, Low: 3, Close: 4, Volume: 5, OpenInt: 6
	Output: outputs data array consisting of entries up to DAYS prior to INITIALDATE for CATEGORY

	"""

	name = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/ETFs/" + name + ".us.txt"
	path = "/Users/sophiasong/Documents/AggregateData/price-volume-data-for-all-us-stocks-etfs/ETFs/*.txt"
	files = glob.glob(path)
	stockData = []



	date = datetime.date(*map(int, initialDate.split('-')))
	dates = []

	for day in range(days+1):
		prevday = str(date - datetime.timedelta(days = day))
		dates.append(prevday)

	for file in files:
		if name == file:
			with open(file) as f:
				csv_reader = csv.reader(f, delimiter = ",")
				for row in csv_reader:
					if row[0] in dates:
						stockData.append(row[category])
	#print(stockData)
	return stockData



def getNewsData(date):
	"""Takes in an initial date (str), outputs news headline on that date"""
	path = "/Users/sophiasong/Documents/AggregateData/us-financial-news-articles/2018_01_11/*.json"
	files = glob.glob(path)
	headlines = []
	for file in files:
		input_file = open(file, 'r')
		json_decode = json.load(input_file)
		blogDate = json_decode["published"][0:10]
		if date == blogDate:
			headlines.append(json_decode["title"])

	#print(headlines)
	return headlines


def getNewsData3(initialDate, category, days):
	""" 
	Input: initialDate (str), data category (str), days (int), 
	Data categories: "title" (news headlines), "text" (news article)
	Output: outputs data array consisting of entries up to DAYS prior to INITIALDATE for CATEGORY

	"""
	path = "/Users/sophiasong/Documents/AggregateData/us-financial-news-articles/2018_01_11/*.json"
	files = glob.glob(path)
	news = []
	

	date = datetime.date(*map(int, initialDate.split('-')))
	dates = []

	for day in range(days+1):
		prevday = str(date - datetime.timedelta(days = day))
		dates.append(prevday)


	for file in files:
		input_file = open(file, 'r')
		json_decode = json.load(input_file)
		blogDate = json_decode["published"][0:10]
		if blogDate in dates:
			news.append(json_decode[category])


	#print(news)
	return news

def getNewsData2017(initialDate, category, days):
	""" 
	Input: initialDate (str), data category (str), days (int), 
	Data categories: "title" (news headlines), "text" (news article)
	Output: outputs data array consisting of entries up to DAYS prior to INITIALDATE for CATEGORY

	"""
	path = "/Users/sophiasong/Documents/AggregateData/666_20170904105517/666_webhose-2015-07_20170904105917/*.json"
	files = glob.glob(path)
	news = []

	date = datetime.date(*map(int, initialDate.split('-')))
	dates = []

	for day in range(days+1):
		prevday = str(date - datetime.timedelta(days = day))
		dates.append(prevday)

	for file in files:
		input_file = open(file, 'r')
		json_decode = json.load(input_file)
		newsDate = json_decode["published"][0:10]
		if newsDate in dates:
			news.append(json_decode[category])
	
	#print(headlines)
	#print(len(headlines))

	return news



def aggregate(initialDate, name, categoryData, categoryNews, days):
	""" 
	Input: initialDate (str), name of stock/etf (str), categoryData (int), categoryNews (str), days (int)
	categoryData: Open: 1, High: 2, Low: 3, Close: 4, Volume: 5, OpenInt: 6 
	categoryNews: "title" (news headlines), "text" (news article)
	Output: outputs aggregated data array consisting of entries up to DAYS prior to INITIALDATE for CATEGORY for stock NAME
	output[0] = stock data; output[1], output[2] = news data 

	"""

	aggregate = []
	etf = getETF5(initialDate, name, categoryData, days)
	news = getNewsData3(initialDate, categoryNews, days)
	news2 = getNewsData2017(initialDate, categoryNews, days)

	aggregate.append(etf)
	aggregate.append(news)
	aggregate.append(news2)

	#print(aggregate)

	return aggregate


#getETF5("2012-11-09", "acim", 1, 4)
#getNewsData3("2018-01-03", "text", 3)
aggregate("2015-07-28", "acim", 1, "title", 2)