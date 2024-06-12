import sys
import pandas as pd
import requests
import time
from datetime import datetime
from tti.indicators import BollingerBands

API_KEY = "e6e5d6ab4b3c415d9c691501ee505e06"
X_CHAIN = "solana"

startAt = "2021-11-29 00:00:00"
endAt = "2024-05-30 00:00:00"
interval = "15m"

def exportTechnicalIndicators():
  address = "So11111111111111111111111111111111111111112"
  addressType = "token"
  
  timeFrom = int(time.mktime(time.strptime(startAt, '%Y-%m-%d %H:%M:%S')))
  timeTo = int(time.mktime(time.strptime(endAt, '%Y-%m-%d %H:%M:%S')))
  
  url = f"https://public-api.birdeye.so/defi/history_price?address={address}&address_type={addressType}&type={interval}&time_from={timeFrom}&time_to={timeTo}"
  headers = {
      "x-chain": X_CHAIN,
      "X-API-KEY": API_KEY
  }
  response = requests.get(url=url, headers=headers).json()
  inputData = response['data']['items']
  
  for item in inputData:
    unixTime = datetime.fromtimestamp(item['unixTime'])
    item['Date'] = unixTime.strftime("%Y-%m-%d")
    item['High'] = item['value']
    item['Low'] = item['value']
    item['close'] = item['value']
    
  dataFrame = pd.DataFrame.from_dict(inputData, orient='columns')
  dataFrame.index = pd.DatetimeIndex(dataFrame['unixTime'])

  dataFrame = dataFrame.drop(columns=['unixTime'])

  dataFrame = dataFrame.iloc[::-1]

  dataFrame.to_csv("price_data.csv", index=False)
  
  # bollingBandsIndicator = BollingerBands(input_data=dataFrame)
  # bollingBandsIndicator.getTiGraph().show()
  
  # simulationData, simulationStatistics, simulationGraph = bollingBandsIndicator.getTiSimulation(close_values=dataFrame[['close']], max_exposure=None, short_exposure_factor=1.5)
  # simulationGraph.show()


if __name__ == "__main__":
  exportTechnicalIndicators()