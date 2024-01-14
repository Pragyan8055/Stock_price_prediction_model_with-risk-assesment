# Stock_price_prediction_model_with-risk-assesment
In this file I have tried to make multiplr models to perform the causal analysis and prediction of Stock data of US closing price, or just closing prices of tickers available on Yahoo Finance. Initially I have used a programme that engages with the user to take the stocks that he wants to put in his portfolio and allocate weightage which were optimized using Monte-Carlo Simulations. But to perform the risk analysis on twitter there was an issue since I did not have any paid access to Twitter data, I tried scraping it from web, but turns out my web browser is quite new hence web scraping became a bit hard, at the end I had to settle for a limited data that was 4 years old but yet I performed analysis of risk on that only on one stock that was Apple. in future I am planning to use web scraping technique using a different browser getting real time data and do Sentimental Analysis and NLP to derive the risk factors which will be engaging with the user and try to add as many economic insights that we caan derive from the data. Besides this I have developed a multitude of models like the LSTM Model, A fourier model, time series model, Various classification and regression models as well, An LSTM+GRU model as well, I also tried using RL based models but they were quite complex, and also they required some libraries that were posing issues. Using an RL model will also help in development of an RL based automated trader which can be used in BackTrader. I have also made some more research, a list of which I have as below 
<ul>
   <li>A simple LSTM Model</li><li>Twitter based LSTM model</li><li>Twitter based regression and classification</li><li>Prophet based model</li><li>An LSTM+GRU based model</li><li>Fourier and TS Analysis</li><li>MA model</li>
</ul>

<h1>Motivation of Twitter based model</h1>

We are try to predict using twitter data because it is getting crucial for markets to tend towards highly liquid and efficient pricing to eliminate the centralised institutiona like banks that provide high inefficiency in terms of less transparency where our capital is getting invested, using the capital to build the infrastructure to provide services, ehich creates unnecessary expense, at the expense of our capital. This creates an necessity for highly efficient and liquid markets where an individual can invest. But a problem with a lot of these markets is the fact that the pricing is not efficient. For a pricing to be efficient it must reflect all the future cashflows in it's pricing. A lot of this efficiency is in the form of noise that is added to these markets in the form of information assymetry that happens the market. This inefficiency is majorly responsible for a lot of firms to exploit it in the form of arbitrages that are formed because of overvaluing the stock or undervaluing it. This noice that causes in efciciency is incorporated in the system via various mediums, earlier it took about 2-3 days to see a visible change in the market via this mass hysteria caused by these sources. Now it's a matter of hours, there are various behavioral factors involved as well but we think that through later. In a weak-form market this mass hysteria reduces our arbitrage as there are various other frictional forces involved as well like- Bid-ask spread that causes the arbitrager to not trade in an interval of price but only after that. The arbitragers, I fnot act as soon as possible in trading, can loose a margin.
A model that allows arbitragers with enough information to predict the inefficient price and the efficient price(here simply with LSTM, but can accompany with more pricing models to improve efficient pricing) can help lock a greater margin of profit than after the market has adapted to the inefficiency.
The one thing the model currently misses is mixing the LSTM with pricing models to give a better look at the efficient price of the stock, but currently I have tried to make the model as efficient as possible, I have made various analysis to help analyse the bullish and the bearish views of the market and made a lot of analysis on time series, moving average and Fourier analysis to derive the causal inferences from the model, I have also made a very simple LSTM model using chat-gpt to get an idea what to do and made furthur models taking inspirations from that model.

<h2>Models for Analysis</h2>
<ol>
   <li>
   <h3>MA and Time Embedded Transformer model</h3>
   In this model I have trid to plot a very basic multi attention layered model that involves Transformers and Time embeddings to get a an approximate of the efficient pricing, and tovisualize the noies in the system, the methodology involves encoding an decoding the data. Although I am quite skeptical of how well it captures the efficiency of prices but it does give a very naive idea that without noises the time embedding model ha learned and captured the backbone of the model and we can see the actual price moves quite a lot that involves the inefficiencies an noises that are added that are beyond the periphery of this system.
      
![Screenshot 2024-01-14 171206](https://github.com/Pragyan8055/Stock_price_prediction_model_with-risk-assesment/assets/126716148/63ac7d65-beee-4d57-96e1-76e9ea6588b7)
      
![Screenshot 2024-01-14 171226](https://github.com/Pragyan8055/Stock_price_prediction_model_with-risk-assesment/assets/126716148/78fee936-d725-4537-8a08-74bc3f28e05f)
   </li>

   <li>
      <h3>Fourier and moving average analysis</h3>
      In the fourier and time series analysis of our portfolio we have tried to make a few models with some parameters and see thow well they fit into the pricing model and see how well they account for the portfolio prices going up or down. We see that a 10 day MA model seemed to mimic the price history in a better manner than a very heavy 200 day model that was very much like the Multi attention transformer and time embedded model. For fourier model we see that 100 component fourier model was best at mimicking the model but I am not sure if the model could be used to predict the efficient pricing system since such a model may deviate heavily some time than efficiently valuate the correct price of the stock portfolio. i have not done individual analysis of stocks in our portfolio as it might take a lot more ime but I certainly have alloted each a well tuned weight and aggregated their contribution to portfolio well aligned with their weightage.
      
![Screenshot 2024-01-14 174522](https://github.com/Pragyan8055/Stock_price_prediction_model_with-risk-assesment/assets/126716148/fce2959f-2f51-4e97-b501-323bdf910e81)

![Screenshot 2024-01-14 174544](https://github.com/Pragyan8055/Stock_price_prediction_model_with-risk-assesment/assets/126716148/e4296a13-29ed-4980-b01f-e6c25895326d)
      
   </li>
   <li>
      <h3> </h3>
   </li>
</ol>
 
