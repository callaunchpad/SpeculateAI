import csv
import glob
import datetime 
import json

def getETF(date):
	"Takes in a date (str) and outputs corresponding data array for all ETFs"

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
	"Takes in a date (str) and outputs corresponding data array for all Stocks"

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
	"Takes in a date (str) and outputs corresponding data for specified ETF name (str) on that date"
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

def getNewsData(date):
	"Takes in an initial date (str), outputs news headline on that date"
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

def getNewsData3(initialDate):
	"Takes in an initial date (str), outputs news headlines for that date and up to three days prior"
	path = "/Users/sophiasong/Documents/AggregateData/us-financial-news-articles/2018_01_11/*.json"
	files = glob.glob(path)
	headlines = []
	date = datetime.date(*map(int, initialDate.split('-')))

	prevday1 = str(date - datetime.timedelta(days =1))
	prevday2 = str(date - datetime.timedelta(days =2))
	prevday3 = str(date - datetime.timedelta(days =3))

	dates = [initialDate, prevday1, prevday2, prevday3]


	for file in files:
		input_file = open(file, 'r')
		json_decode = json.load(input_file)
		blogDate = json_decode["published"][0:10]
		if blogDate in dates:
			headlines.append(json_decode["title"])


	#print(len(headlines))
	return headlines

def getNewsData2017(initialDate):
	"Takes in an initial date (str), outputs news headline for that date and up to three days prior"
	path = "/Users/sophiasong/Documents/AggregateData/666_20170904105517/666_webhose-2015-07_20170904105917/*.json"
	files = glob.glob(path)
	headlines = []
	fullText = []
	date = datetime.date(*map(int, initialDate.split('-')))

	prevday1 = str(date - datetime.timedelta(days =1))
	prevday2 = str(date - datetime.timedelta(days =2))
	prevday3 = str(date - datetime.timedelta(days =3))

	dates = [initialDate, prevday1, prevday2, prevday3]

	for file in files:
		input_file = open(file, 'r')
		json_decode = json.load(input_file)
		newsDate = json_decode["published"][0:10]
		if newsDate in dates:
			headlines.append(json_decode["title"])
			fullText.append(json_decode["text"])
	
	#print(headlines)
	#print(len(headlines))

	return headlines 






def aggregate(date, name):
	aggregate = []
	etf = getETF5(date, name)
	news = getNewsData2017(date)

	aggregate.append(etf)
	aggregate.append(news)

	print(aggregate)

	return aggregate




#getETF5("2012-03-08", "acim")
#getNewsData2017("2015-07-28")
aggregate("2015-07-28", "acim")
