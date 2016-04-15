from Stock import *
from Utils import *
import time
import gc
import thread
import random
import os

ANALYSIS_TYPE = 'short'  # 'long'
RS_THS = 0.7
now = datetime.now()
EXTENDED_DEBUG = False
DEBUG_CONDITIONS = True

class IntersectBasedAnalysisClass:

    stocksList = []
    numStocksInList = 0
    stock = StockClass()
    stocks4Analysis = []
    erroneousStocks = []
    out_file = 0

    def getStocksList(self, i_update=False, i_listOrigin='NASDAQ', i_debug=False):
        if (i_update):
            refreshStocksList()
        if (i_listOrigin == 'NASDAQ'):
            listOfStocks = readFileContent('NASDAQ', '|', 0)
            listOfStocks.drop(listOfStocks.tail(1).index, inplace=True)
            self.numStocksInList = len(listOfStocks['Symbol'])
            self.stocksList = random.sample(listOfStocks['Symbol'], self.numStocksInList)
        else:
            listOfStocks = readFileContent('OTHERS', '|', 0)
            listOfStocks.drop(listOfStocks.tail(1).index, inplace=True)
            self.numStocksInList = len(listOfStocks['ACT Symbol'])
            self.stocksList = random.sample(listOfStocks['ACT Symbol'], self.numStocksInList)

        # self.stocksList = ['SSP']
        # self.numStocksInList = 1
        if i_debug:
            if EXTENDED_DEBUG:
                print i_listOrigin, self.numStocksInList
                print self.stocksList
                # out_file.write(self.numStocksInList)
                # out_file.write(i_listOrigin)
                # out_file.write(self.stocksList)

    def getSpyData(self):
        if EXTENDED_DEBUG:
            print "#### Start acquisitions of SPY ####"
            self.out_file.write("#### Start acquisitions of SPY ####\n")
        self.stock.getData(i_symbol='SPY', i_destDictKey='SPY')
        self.stock.getMovementType(i_destDictKey='SPY')
        self.stock.reversalPointsDetector(i_destDictKey='SPY')
        if EXTENDED_DEBUG:
            print "#### End acquisitions of SPY ####"
            out_file.write("#### End acquisitions of SPY ####\n")

    def checkIfUpdate(self):
        # day = datetime.today().day
        lastEntryDate = self.stock.getDataDate()
        print lastEntryDate
        self.out_file.write("Last entry's day: %d/%d\n" % (lastEntryDate.month, lastEntryDate.day))
        # if (day == lastEntryDate.day):
        #     return True
        # else:
        #     return False

    def getData(self):
        time.sleep(2)
        if EXTENDED_DEBUG:
            print "#### Start acquisitions of SPY ####"
            self.out_file.write("#### Start acquisitions of SPY ####\n")
        self.stock.getData(i_symbol='SPY', i_destDictKey='SPY')
        if EXTENDED_DEBUG:
            print "#### End acquisitions of SPY ####"
            self.out_file.write("#### End acquisitions of SPY ####\n")
        idx = 0

        for symbolName in self.stocksList:
            # stock = Stock(name=symbolName)
            self.stock.__init__(name=symbolName)

            # get data of required symbol
            idx = idx + 1
            if EXTENDED_DEBUG:
                print '#### [', idx, '/', self.numStocksInList, ']: Start acquisitions of [', symbolName, '] ####'
                self.out_file.write("#### [ %d / %d ]: Start acquisitions of [ %s ] ####\n" % (idx, self.numStocksInList, symbolName))
            else:
                print '[', idx, '/', self.numStocksInList, ']'
                self.out_file.write("[ %d / %d ]\n" % (idx, self.numStocksInList))
            try:
                self.stock.getData(i_symbol=symbolName, i_destDictKey='symbol')
            except:
                self.erroneousStocks.append(symbolName)
                save_obj(self.erroneousStocks, 'erroneousStocks_' + ANALYSIS_TYPE)
                if EXTENDED_DEBUG:
                    print '!!!! GetData ERROR !!!!'
                    self.out_file.write('!!!! GetData ERROR !!!!\n')
                continue


            if EXTENDED_DEBUG:
                print '#### End acquisitions of [', symbolName, '] ####'
                self.out_file.write("#### End acquisitions of [ %s ] ####\n" % symbolName)

    def restoreSymbol(self, i_symbol):
        self.stocks4Analysis = load_obj(i_symbol)

    def main(self):
        while True:
            dayOfWeek = datetime.today().weekday()
            hour = datetime.today().hour
            minute = datetime.today().minute
            if (dayOfWeek >= 1) and (dayOfWeek <= 5) and ((hour+3) == 14) and (minute == 00):
            # if (1):
                self.out_file = open('output_'+str(now.day)+'_'+str(now.month)+'_'+str(now.year)+'_'+str(now.hour)+'.txt', "w")
                self.getSpyData()
                self.checkIfUpdate()
                self.getStocksList(i_listOrigin='NASDAQ', i_debug=True)
                self.getData()
                self.getStocksList(i_listOrigin='OTHERS', i_debug=True)
                self.getData()
                self.out_file.close()
            else:
                print 'DaylOfWeek: ', dayOfWeek, ' hour: ', hour+3, ' minute: ', minute, 'sleep 60s - waiting...'
                time.sleep(60)

# ----------------- Main program -------------------
#os.system("taskkill /im python.exe")
#os.system("taskkill /im python.exe")
#os.system("taskkill /im python.exe")
isBaseAnalysis = IntersectBasedAnalysisClass()
isBaseAnalysis.main()
# isBaseAnalysis.restoreSymbol('stocks4Analysis')
