import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import streamlit as st
st.set_page_config(layout="wide")
from datetime import datetime
from dateutil.relativedelta import relativedelta
#from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
#import scipy.optimize as optimize
#from PIL import Image
import cvxpy as cp
import json
from streamlit_lottie import st_lottie


metrics = "bondMetrics.csv"


# Plotly graph themes
theme='seaborn'



#allBonds = ['1Y', '4Y', '7Y', '12Y', '20Y', '30Y']


redemption=1

@st.cache_data
def nperiods(ncd, maturity):
    n = round((((maturity - ncd).days)/(365.25/2)),0)
    return n

@st.cache_data
def cumex(s, bcd):
    if(s < bcd):
        return 1
    else:
        return 0

@st.cache_data    
def daysacc(s, lcd, ncd, cum):
    #cum = cumex(s, bcd)

    if (cum == 1):
        days = (s - lcd).days
    else:
        days = (s-ncd).days
    return days

@st.cache_data
def couponPay(cb, cum):
    cpn=(cb/2)
    #cum = cumex(s, bcd)
    cpnNCD = (cpn)*cum

    return cpnNCD

@st.cache_data
def factor(yieldm):
    f = (1+(yieldm/2))**-1
    return f

@st.cache_data
def brokenPeriod(s, lcd, ncd, maturity):
    if(ncd == maturity):
        bp = (ncd-s)/(365/2)
    else:
        bp = (ncd-s)/(ncd-lcd)
    
    return bp

@st.cache_data
def brokenPeriodDF(f, bp, ncd, maturity):
    if(ncd==maturity):
        bpf = f/(f+(bp*(1-f)))
    else:
        bpf = f**bp

    return bpf

@st.cache_data
def accint(daysacc, cb):
    accint = (daysacc*cb)/365
    return accint

@st.cache_data
def accintRound(daysacc, cb):
    accint = round((daysacc*cb)/365,7)
    return accint

@st.cache_data
def aip(n, bpf,cpnncd, cb, f, r):

    cpn=(cb/2)

    if(f==1):
        price=cpnncd+(cpn*n) + r
    else:
        price = bpf*(cpnncd+(cpn*((f*(1-(f**n)))/(1-f))+(r*(f**n))))

    return price


@st.cache_data
def clean(aip, accint):
    cleanPrice = (aip - accint)
    return cleanPrice

@st.cache_data
def cleanRound(aip, accint):
    cleanPrice = round((aip - accint),7)
    return cleanPrice


@st.cache_data
def aipRound(cpRound, accintRound):
    price = (cpRound + accintRound)
    return price


@st.cache_data
def dbpf(f, bp, bpf, ncd, maturity):
    if(ncd==maturity):
        dbpf = (bp*(bpf**2))/(f**2)
    else:
        dbpf = (bp*bpf)/f

    return dbpf


@st.cache_data
def d2bpf(f, bp, bpf, dbpf, ncd, maturity):
    if(ncd==maturity):
        d2bpf = (2*dbpf)*((bp*bpf - f)/(f**2))
    else:
        d2bpf = dbpf*((bp-1)/f)

    return d2bpf


@st.cache_data
def dcpn(n, cb, f):
    cpn = cb/2

    if(f==1):
        dcpn=cpn*((n*(n+1))/2)
    else:
        dcpn=cpn*((1-(n-(n*f)+1)*(f**n))/((1-f)**2))
    
    return dcpn


@st.cache_data
def d2cpn(n, cb, f):
    cpn=cb/2

    if(f==1):
        d2cpn = cpn*((n*((n**2)-1))/3)
    else:
        d2cpn = cpn*((2-((n*(1-f)*(2+(n-1)*(1-f))+(2*f))*(f**(n-1))))/((1-f)**3))

    return d2cpn

@st.cache_data
def dr(n,f):
    dr=n*1*(f**(n-1))
    return dr

@st.cache_data
def d2r(n,f):
    d2r=n*(n-1)*1*(f**(n-2))
    return d2r



@st.cache_data
def daip(dbpf, aip, bpf, dcpn, dr):

    daip = dbpf*(aip/bpf) + bpf*(dcpn+dr)

    return daip


@st.cache_data
def d2aip(d2bpf, aip, bpf, dbpf, daip, dcpn, d2cpn, dr, d2r):

    d2aip = d2bpf*(aip/bpf) + dbpf*(((bpf*daip - aip*dbpf)/(bpf**2))+dcpn+dr)+bpf*(d2cpn+d2r)

    return d2aip



@st.cache_data
def delta(f, daip, dy):

    delta = 1*((f**2)/200)*(daip/(dy))

    return delta



@st.cache_data
def randperbp(delta):

    rpbp = 100*delta

    return rpbp


#@st.cache_data
#def dmod(aip, daip, dy):
#
#    dmod = -100*(daip/dy)/aip
#
#    return dmod

######################################################################################################################
@st.cache_data
def modfinal(s, lcd, ncd, bcd, maturity, cb, yieldm, dy=1):

    aip = allInPrice(s, lcd, ncd, bcd, maturity, cb, yieldm)

    cpn = cb/2

    f = (1+(yieldm/2))**-1

    n = round((((maturity - ncd).days)/(365.25/2)),0)


    if(ncd == maturity):
        bp = (ncd-s)/(365/2)
    else:
        bp = (ncd-s)/(ncd-lcd)


    if(ncd==maturity):
        bpf = f/(f+(bp*(1-f)))
    else:
        bpf = f**bp


    
    if(ncd==maturity):
        dbpf = (bp*(bpf**2))/(f**2)
    else:
        dbpf = (bp*bpf)/f

    ##if(ncd==maturity):
    ##    d2bpf = (2*dbpf)*((bp*bpf - f)/(f**2))
    ##else:
    ##    d2bpf = dbpf*((bp-1)/f)


    if(f==1):
        dcpn=cpn*((n*(n+1))/2)
    else:
        dcpn=cpn*((1-(n-(n*f)+1)*(f**n))/((1-f)**2))


    ##if(f==1):
    ##    d2cpn = cpn*((n*((n**2)-1))/3)
    ##else:
    ##    d2cpn = cpn*((2-((n*(1-f)*(2+(n-1)*(1-f))+(2*f))*(f**(n-1))))/((1-f)**3))


    dr=n*1*(f**(n-1))

    #d2r=n*(n-1)*1*(f**(n-2))

    daip = dbpf*(aip/bpf) + bpf*(dcpn+dr)

    #d2aip = d2bpf*(aip/bpf) + dbpf*(((bpf*daip - aip*dbpf)/(bpf**2))+dcpn+dr)+bpf*(d2cpn+d2r)

    delta = 1*((f**2)/200)*(daip/(dy))

    dmod = 100*(delta/aip)

    ##rpbp = 100*delta

    return dmod



#########################################################################################################################

@st.cache_data
def durationfinal(s, lcd, ncd, bcd, maturity, cb, yieldm, dy=1):

    aip = allInPrice(s, lcd, ncd, bcd, maturity, cb, yieldm)

    cpn = cb/2

    f = (1+(yieldm/2))**-1

    n = round((((maturity - ncd).days)/(365.25/2)),0)


    if(ncd == maturity):
        bp = (ncd-s)/(365/2)
    else:
        bp = (ncd-s)/(ncd-lcd)


    if(ncd==maturity):
        bpf = f/(f+(bp*(1-f)))
    else:
        bpf = f**bp


    
    if(ncd==maturity):
        dbpf = (bp*(bpf**2))/(f**2)
    else:
        dbpf = (bp*bpf)/f

    ##if(ncd==maturity):
    ##    d2bpf = (2*dbpf)*((bp*bpf - f)/(f**2))
    ##else:
    ##    d2bpf = dbpf*((bp-1)/f)


    if(f==1):
        dcpn=cpn*((n*(n+1))/2)
    else:
        dcpn=cpn*((1-(n-(n*f)+1)*(f**n))/((1-f)**2))


    ##if(f==1):
    ##    d2cpn = cpn*((n*((n**2)-1))/3)
    ##else:
    ##    d2cpn = cpn*((2-((n*(1-f)*(2+(n-1)*(1-f))+(2*f))*(f**(n-1))))/((1-f)**3))


    dr=n*1*(f**(n-1))

    #d2r=n*(n-1)*1*(f**(n-2))

    daip = dbpf*(aip/bpf) + bpf*(dcpn+dr)

    #d2aip = d2bpf*(aip/bpf) + dbpf*(((bpf*daip - aip*dbpf)/(bpf**2))+dcpn+dr)+bpf*(d2cpn+d2r)

    delta = 1*((f**2)/200)*(daip/(dy))

    dmod = 100*(delta/aip)

    dur = dmod/f

    ##rpbp = 100*delta

    return dur




####################################################################################################################

@st.cache_data
def deltafinal(s, lcd, ncd, bcd, maturity, cb, yieldm, dy=1):

    aip = allInPrice(s, lcd, ncd, bcd, maturity, cb, yieldm)

    cpn = cb/2

    f = (1+(yieldm/2))**-1

    n = round((((maturity - ncd).days)/(365.25/2)),0)


    if(ncd == maturity):
        bp = (ncd-s)/(365/2)
    else:
        bp = (ncd-s)/(ncd-lcd)


    if(ncd==maturity):
        bpf = f/(f+(bp*(1-f)))
    else:
        bpf = f**bp


    
    if(ncd==maturity):
        dbpf = (bp*(bpf**2))/(f**2)
    else:
        dbpf = (bp*bpf)/f

    ##if(ncd==maturity):
    ##    d2bpf = (2*dbpf)*((bp*bpf - f)/(f**2))
    ##else:
    ##    d2bpf = dbpf*((bp-1)/f)


    if(f==1):
        dcpn=cpn*((n*(n+1))/2)
    else:
        dcpn=cpn*((1-(n-(n*f)+1)*(f**n))/((1-f)**2))


    ##if(f==1):
    ##    d2cpn = cpn*((n*((n**2)-1))/3)
    ##else:
    ##    d2cpn = cpn*((2-((n*(1-f)*(2+(n-1)*(1-f))+(2*f))*(f**(n-1))))/((1-f)**3))


    dr=n*1*(f**(n-1))

    #d2r=n*(n-1)*1*(f**(n-2))

    daip = dbpf*(aip/bpf) + bpf*(dcpn+dr)

    #d2aip = d2bpf*(aip/bpf) + dbpf*(((bpf*daip - aip*dbpf)/(bpf**2))+dcpn+dr)+bpf*(d2cpn+d2r)

    delta = 1*((f**2)/200)*(daip/(dy))*(-100)

    ##dmod = 100*(delta/aip)

    ##rpbp = 100*delta

    return delta




####################################################################################################################
@st.cache_data
def rpbpfinal(s, lcd, ncd, bcd, maturity, cb, yieldm, dy=1):

    aip = allInPrice(s, lcd, ncd, bcd, maturity, cb, yieldm)

    cpn = cb/2

    f = (1+(yieldm/2))**-1

    n = round((((maturity - ncd).days)/(365.25/2)),0)


    if(ncd == maturity):
        bp = (ncd-s)/(365/2)
    else:
        bp = (ncd-s)/(ncd-lcd)


    if(ncd==maturity):
        bpf = f/(f+(bp*(1-f)))
    else:
        bpf = f**bp


    
    if(ncd==maturity):
        dbpf = (bp*(bpf**2))/(f**2)
    else:
        dbpf = (bp*bpf)/f

    ##if(ncd==maturity):
    ##    d2bpf = (2*dbpf)*((bp*bpf - f)/(f**2))
    ##else:
    ##    d2bpf = dbpf*((bp-1)/f)


    if(f==1):
        dcpn=cpn*((n*(n+1))/2)
    else:
        dcpn=cpn*((1-(n-(n*f)+1)*(f**n))/((1-f)**2))


    ##if(f==1):
    ##    d2cpn = cpn*((n*((n**2)-1))/3)
    ##else:
    ##    d2cpn = cpn*((2-((n*(1-f)*(2+(n-1)*(1-f))+(2*f))*(f**(n-1))))/((1-f)**3))


    dr=n*1*(f**(n-1))

    #d2r=n*(n-1)*1*(f**(n-2))

    daip = dbpf*(aip/bpf) + bpf*(dcpn+dr)

    #d2aip = d2bpf*(aip/bpf) + dbpf*(((bpf*daip - aip*dbpf)/(bpf**2))+dcpn+dr)+bpf*(d2cpn+d2r)

    delta = 1*((f**2)/200)*(daip/(dy))

    #dmod = 100*(delta/aip)

    rpbp = 10000*delta

    return rpbp

#################################################################################################################
@st.cache_data
def convexityfinal(s, lcd, ncd, bcd, maturity, cb, yieldm, dy=1):

    aip = allInPrice(s, lcd, ncd, bcd, maturity, cb, yieldm)

    cpn = cb/2

    f = (1+(yieldm/2))**-1

    n = round((((maturity - ncd).days)/(365.25/2)),0)


    if(ncd == maturity):
        bp = (ncd-s)/(365/2)
    else:
        bp = (ncd-s)/(ncd-lcd)


    if(ncd==maturity):
        bpf = f/(f+(bp*(1-f)))
    else:
        bpf = f**bp


    
    if(ncd==maturity):
        dbpf = (bp*(bpf**2))/(f**2)
    else:
        dbpf = (bp*bpf)/f

    if(ncd==maturity):
        d2bpf = (2*dbpf)*((bp*bpf - f)/(f**2))
    else:
        d2bpf = dbpf*((bp-1)/f)


    if(f==1):
        dcpn=cpn*((n*(n+1))/2)
    else:
        dcpn=cpn*((1-(n-(n*f)+1)*(f**n))/((1-f)**2))


    if(f==1):
        d2cpn = cpn*((n*((n**2)-1))/3)
    else:
        d2cpn = cpn*((2-((n*(1-f)*(2+(n-1)*(1-f))+(2*f))*(f**(n-1))))/((1-f)**3))


    dr=n*1*(f**(n-1))

    d2r=n*(n-1)*1*(f**(n-2))

    daip = dbpf*(aip/bpf) + bpf*(dcpn+dr)

    d2aip = d2bpf*(aip/bpf) + dbpf*(((bpf*daip - aip*dbpf)/(bpf**2))+dcpn+dr)+bpf*(d2cpn+d2r)



    diff2 = ((((daip/dy)*(f**3))/2) + (((d2aip/(dy**2))*(f**4))/4))/10000


    conv = (10000/aip)*diff2
    ##delta = 1*((f**2)/200)*(daip/(dy))

    ##dmod = 100*(delta/aip)

    ##rpbp = 100*delta

    return conv



################################################################################################################

@st.cache_data
def dmod(delta,aip):
    dmod = 100*(delta/aip)
    return dmod


@st.cache_data
def seconddiff(daip, d2aip, f, dy):
    diff2 = ((((daip/dy)*(f**3))/2) + (((d2aip/(dy**2))*(f**4))/4))/10000
    return diff2



#@st.cache_data
#def conv(aip, d2aip, dy):
#
#    conv = (10000/aip)*(d2aip/(dy**2))
#
#    return conv

@st.cache_data
def conv(aip, diff2):
    conv = (10000/aip)*diff2
    return conv

#########################################

@st.cache_data
def allInPrice(s, lcd, ncd, bcd, maturity, coupon, yieldm):

    n = round((((maturity - ncd).days)/(365.25/2)),0)

    if(s < bcd):
        cum = 1
    else:
        cum = 0


    if (cum == 1):
        daysacc = (s - lcd).days
    else:
        daysacc = (s-ncd).days


    cpn=(coupon/2)
    
    cpnNCD = (cpn)*cum

    f = (1+(yieldm/2))**-1


    if(ncd == maturity):
        bp = (ncd-s)/(365/2)
    else:
        bp = (ncd-s)/(ncd-lcd)


    if(ncd==maturity):
        bpf = f/(f+(bp*(1-f)))
    else:
        bpf = f**bp


    accint = (daysacc*coupon)/365

    accintRound = round((daysacc*coupon)/365,7)


    if(f==1):
        price=cpnNCD+(cpn*n) + 1
    else:
        price = bpf*(cpnNCD+(cpn*((f*(1-(f**n)))/(1-f))+(1*(f**n))))


    #cleanPrice = (aip - accint)

    cleanPriceRound = round((price - accint),7)

    priceRound = (cleanPriceRound  + accintRound)

    return priceRound
    




########################################



#check1, check2 = st.columns([1,1])

#banner1 = Image.open('smbanner.jpg')
#check2.image(banner1, width=600)


#st.header('Bond Analysis')
#st.markdown("<h1 style='text-align: left; color: charcoal; padding-left: 0px; font-size: 40px'><b>""<b></h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: left; color: charcoal; padding-left: 0px; font-size: 40px'><b>Bond Analysis<b></h1>", unsafe_allow_html=True)

#st.image(banner1, width=600)
#st.markdown("<i style='text-align: left; color: charcoal; padding-left: 0px; font-size: 15px'><b>JSE Bond Pricer Link - https://bondcalculator.jse.co.za/BondSingle.aspx?calc=Spot<b></i>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: left; color: darkred; padding-left: 0px; font-size: 25px'><b>Bond Metrics<b></h1>", unsafe_allow_html=True)

#st.sidebar.header('User Inputs')
#st.sidebar.markdown("<h1 style='text-align: left; color: charcoal; padding-left: 0px; font-size: 40px'><b>""<b></h1>", unsafe_allow_html=True)
#st.sidebar.markdown("<h1 style='text-align: left; color: charcoal; padding-left: 0px; font-size: 40px'><b>User Inputs<b></h1>", unsafe_allow_html=True)




url='data.json'

with open(url, 'r') as fson:  
    res = json.load(fson)


url_json = res


with st.sidebar:
    st_lottie(url_json,
            # change the direction of our animation
            reverse=True,
            # height and width of animation
            height=200,  
            width=200,
            # speed of animation
            speed=1,  
            # means the animation will run forever like a gif, and not as a still image
            loop=True,  
            # quality of elements used in the animation, other values are "low" and "medium"
            quality='high',
            # THis is just to uniquely identify the animation
            key='banner',
            )





st.sidebar.markdown("<h1 style='text-align: left; color: charcoal; padding-left: 0px; font-size: 40px'><b>User Inputs<b></h1>", unsafe_allow_html=True)




#selected_bonds = st.multiselect('Bond Selection', allBonds, default=allBonds)
#bonds = selected_bonds
#bonds = allBonds

today = datetime.today().date()

lastCouponDate = today + relativedelta(months=-6)
tbBookClose = today + relativedelta(days=-10)
tbRedemption = today + relativedelta(months=+12)

#st.markdown(today)
#st.markdown(lastCouponDate)
#st.markdown(tbBookClose)
#st.markdown(tbRedemption)

#settleDate = st.date_input('Settlment Date', today)
st.sidebar.markdown(' ')
#st.sidebar.markdown(' ')
#st.sidebar.markdown('Bond Settlement')
st.sidebar.markdown("<h1 style='text-align: left; color: #008080; padding-left: 0px; font-size: 20px'><b>Bond Settlement<b></h1>", unsafe_allow_html=True)
settleDate = st.sidebar.date_input('Settlment Date', today)



#settleDate = datetime(2005, 8, 26).date()


@st.cache_data
def dataNew(file):
    #df = pd.read_csv(file, delimiter=";", skipinitialspace = True)
    df = pd.read_csv(file, delimiter=",", skipinitialspace = True)
    return df





##benchmark_input = st.sidebar.number_input('Repo-Rate', min_value=0.00, max_value=20.00, format='%0.2f', step=0.25, value=8.25)
##benchmark=benchmark_input/100
benchmark=0.0825


#target_input = st.sidebar.number_input('Target-Rate', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=11.00)
#target=target_input/100

metricsDF = dataNew(metrics)

metricsDF.index=metricsDF['Period']
metricsDF.drop(['Period'], axis=1, inplace=True)

#R186yield = metricsDF['Yield'][0]
#R2030yield = metricsDF['Yield'][1]
#R2035yield = metricsDF['Yield'][2]
#R2044yield = metricsDF['Yield'][3]
#R2053yield = metricsDF['Yield'][4]

#R186MD = metricsDF['MD'][0]
#R2030MD = metricsDF['MD'][1]
#R2035MD = metricsDF['MD'][2]
#R2044MD = metricsDF['MD'][3]
#R2053MD = metricsDF['MD'][4]

#R186Con = metricsDF['Convexity'][0]
#R2030Con = metricsDF['Convexity'][1]
#R2035Con = metricsDF['Convexity'][2]
#R2044Con = metricsDF['Convexity'][3]
#R2053Con = metricsDF['Convexity'][4]


st.sidebar.markdown(' ')
#st.sidebar.markdown(' ')
#st.sidebar.markdown('Bond Yields')
st.sidebar.markdown("<h1 style='text-align: left; color: #008080; padding-left: 0px; font-size: 20px'><b>Bond Yields<b></h1>", unsafe_allow_html=True)
RtbYieldAdj = st.sidebar.number_input('1Y - TB (%)', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=metricsDF['Yield'][0]*100)
R186YieldAdj = st.sidebar.number_input('4Y - R186 (%)', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=metricsDF['Yield'][1]*100)
R2030YieldAdj = st.sidebar.number_input('7Y - R2030 (%)', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=metricsDF['Yield'][2]*100)
R2035YieldAdj = st.sidebar.number_input('12Y - R2035 (%)', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=metricsDF['Yield'][3]*100)
R2044YieldAdj = st.sidebar.number_input('20Y - R2044 (%)', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=metricsDF['Yield'][4]*100)
R2053YieldAdj = st.sidebar.number_input('30Y - R2053 (%)', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=metricsDF['Yield'][5]*100)

#st.sidebar.markdown('MD')
#R186MDAdj = st.sidebar.number_input('R186 MD', min_value=0.00, max_value=10.00, format='%0.2f', step=0.25, value=metricsDF['MD'][0]*100)
#R2030MDAdj = st.sidebar.number_input('R2030 MD', min_value=0.00, max_value=10.00, format='%0.2f', step=0.25, value=metricsDF['MD'][1]*100)
#R2035MDAdj = st.sidebar.number_input('R2035 MD', min_value=0.00, max_value=10.00, format='%0.2f', step=0.25, value=metricsDF['MD'][2]*100)
#R2044MDAdj = st.sidebar.number_input('R2044 MD', min_value=0.00, max_value=10.00, format='%0.2f', step=0.25, value=metricsDF['MD'][3]*100)
#R2053MDAdj = st.sidebar.number_input('R2053 MD', min_value=0.00, max_value=10.00, format='%0.2f', step=0.25, value=metricsDF['MD'][3]*100)


#st.sidebar.markdown('Convexity')
#R186ConAdj = st.sidebar.number_input('R186 MD', min_value=0.00, max_value=1000.00, format='%0.2f', step=0.25, value=metricsDF['Convexity'][0]*100)
#R2030ConAdj = st.sidebar.number_input('R2030 MD', min_value=0.00, max_value=1000.00, format='%0.2f', step=0.25, value=metricsDF['Convexity'][1]*100)
#R2035ConAdj = st.sidebar.number_input('R2035 MD', min_value=0.00, max_value=1000.00, format='%0.2f', step=0.25, value=metricsDF['Convexity'][2]*100)
#R2044ConAdj = st.sidebar.number_input('R2044 MD', min_value=0.00, max_value=1000.00, format='%0.2f', step=0.25, value=metricsDF['Convexity'][3]*100)
#R2053ConAdj = st.sidebar.number_input('R2053 MD', min_value=0.00, max_value=1000.00, format='%0.2f', step=0.25, value=metricsDF['Convexity'][4]*100)


metricsDF['Yield'][0] = RtbYieldAdj/100
metricsDF['Yield'][1] = R186YieldAdj/100
metricsDF['Yield'][2] = R2030YieldAdj/100
metricsDF['Yield'][3] = R2035YieldAdj/100
metricsDF['Yield'][4] = R2044YieldAdj/100
metricsDF['Yield'][5] = R2053YieldAdj/100

#metricsDF['MD'][0] = R186MDAdj/100
#metricsDF['MD'][1] = R2030MDAdj/100
#metricsDF['MD'][2] =R2035MDAdj/100
#metricsDF['MD'][3] =R2044MDAdj/100
#metricsDF['MD'][4] = R2053MDAdj/100

#metricsDF['Convexity'][0] = R186ConAdj/100
#metricsDF['Convexity'][1] = R2030ConAdj/100
#metricsDF['Convexity'][2] = R2035ConAdj/100
#metricsDF['Convexity'][3] = R2044ConAdj/100
#metricsDF['Convexity'][4] = R2053ConAdj/100


metricsDF['LCD'] = metricsDF.apply(lambda x: datetime.strptime(x['LCD'], "%Y/%m/%d").date(), axis=1)
metricsDF['NCD'] = metricsDF.apply(lambda x: datetime.strptime(x['NCD'], "%Y/%m/%d").date(), axis=1)
metricsDF['BCD'] = metricsDF.apply(lambda x: datetime.strptime(x['BCD'], "%Y/%m/%d").date(), axis=1)
metricsDF['Maturity'] = metricsDF.apply(lambda x: datetime.strptime(x['Maturity'], "%Y/%m/%d").date(), axis=1)

metricsDF['LCD'][0] = lastCouponDate
metricsDF['NCD'][0] = today
metricsDF['BCD'][0] = tbBookClose
metricsDF['Maturity'][0] = tbRedemption

#metricsDF['LCDdays'] = metricsDF.apply(lambda x: ((x['LCD'] - today).days)/365, axis=1)
#metricsDF['NCDdays'] = metricsDF.apply(lambda x: ((x['NCD'] - today).days)/365, axis=1)
#metricsDF['Term'] = metricsDF.apply(lambda x: ((x['Maturity'] - today).days)/365, axis=1)




#########################################

metricsDF['nPeriods'] = metricsDF.apply(lambda x: nperiods(x['NCD'], x['Maturity']), axis=1)
metricsDF['Cumex'] = metricsDF.apply(lambda x: cumex(settleDate, x['BCD']), axis=1)
metricsDF['DaysAcc'] = metricsDF.apply(lambda x: daysacc(settleDate, x['LCD'], x['NCD'], x['Cumex']), axis=1)
metricsDF['CouponPay'] = metricsDF.apply(lambda x: couponPay(x['Coupon'], x['Cumex']), axis=1)
metricsDF['Factor'] = metricsDF.apply(lambda x: factor(x['Yield']), axis=1)
metricsDF['BrokenPeriod'] = metricsDF.apply(lambda x: brokenPeriod(settleDate, x['LCD'], x['NCD'], x['Maturity']), axis=1)
metricsDF['AccInt'] = metricsDF.apply(lambda x: accint(x['DaysAcc'], x['Coupon']), axis=1)
metricsDF['AccIntRound'] = metricsDF.apply(lambda x: accintRound(x['DaysAcc'], x['Coupon']), axis=1)
metricsDF['BrokenPeriodDF'] = metricsDF.apply(lambda x: brokenPeriodDF(x['Factor'], x['BrokenPeriod'], x['NCD'], x['Maturity']), axis=1)
metricsDF['AIP0'] = metricsDF.apply(lambda x: aip(x['nPeriods'], x['BrokenPeriodDF'], x['CouponPay'], x['Coupon'], x['Factor'], 1), axis=1)
metricsDF['Clean'] = metricsDF.apply(lambda x: clean(x['AIP0'], x['AccInt']), axis=1)
metricsDF['CleanRound'] = metricsDF.apply(lambda x: cleanRound(x['AIP0'], x['AccInt']), axis=1)
metricsDF['AIP0Round'] = metricsDF.apply(lambda x: aipRound(x['CleanRound'], x['AccIntRound']), axis=1)
metricsDF['AllInPrice'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], x['Yield']), axis=1)


metricsDF['dbpf'] = metricsDF.apply(lambda x: dbpf(x['Factor'], x['BrokenPeriod'], x['BrokenPeriodDF'], x['NCD'], x['Maturity']), axis=1)

metricsDF['d2bpf'] = metricsDF.apply(lambda x: d2bpf(x['Factor'], x['BrokenPeriod'], x['BrokenPeriodDF'], x['dbpf'], x['NCD'], x['Maturity']), axis=1)

metricsDF['dcpn'] = metricsDF.apply(lambda x: dcpn(x['nPeriods'], x['Coupon'], x['Factor']), axis=1)

metricsDF['d2cpn'] = metricsDF.apply(lambda x: d2cpn(x['nPeriods'], x['Coupon'], x['Factor']), axis=1)

metricsDF['dr'] = metricsDF.apply(lambda x: dr(x['nPeriods'], x['Factor']), axis=1)

metricsDF['d2r'] = metricsDF.apply(lambda x: d2r(x['nPeriods'], x['Factor']), axis=1)

metricsDF['daip'] = metricsDF.apply(lambda x: daip(x['dbpf'], x['AllInPrice'], x['BrokenPeriodDF'], x['dcpn'], x['dr']), axis=1)

metricsDF['d2aip'] = metricsDF.apply(lambda x: d2aip(x['d2bpf'], x['AllInPrice'], x['BrokenPeriodDF'], x['dbpf'], x['daip'], x['dcpn'], x['d2cpn'], x['dr'], x['d2r']), axis=1)


metricsDF['deltapy'] = metricsDF.apply(lambda x: delta(x['Factor'], x['daip'], dy=1), axis=1)
metricsDF['rpbppy'] = metricsDF.apply(lambda x: randperbp(x['deltapy']), axis=1)
#metricsDF['dmodpy'] = metricsDF.apply(lambda x: dmod(x['AllInPrice'], x['daip'], dy=0.0001), axis=1)
metricsDF['dmodpy'] = metricsDF.apply(lambda x: dmod(x['deltapy'],x['AllInPrice']), axis=1)
metricsDF['seconddiff'] = metricsDF.apply(lambda x: seconddiff(x['daip'],x['d2aip'], x['Factor'], dy=1), axis=1)
metricsDF['convpy'] = metricsDF.apply(lambda x: conv(x['AllInPrice'],x['seconddiff']), axis=1)


metricsDF['DurationFunc'] = metricsDF.apply(lambda x: durationfinal(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], x['Yield']), axis=1)
metricsDF['MDFunc'] = metricsDF.apply(lambda x: modfinal(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], x['Yield']), axis=1)
metricsDF['DeltaFunc'] = metricsDF.apply(lambda x: deltafinal(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], x['Yield']), axis=1)
metricsDF['RPBPFunc'] = metricsDF.apply(lambda x: rpbpfinal(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], x['Yield']), axis=1)
metricsDF['ConvexityFunc'] = metricsDF.apply(lambda x: convexityfinal(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], x['Yield']), axis=1)
#metricsDF['convpy'] = metricsDF.apply(lambda x: conv(x['AllInPrice'], x['d2aip'], dy=0.01), axis=1)

#allInPrice(s, lcd, ncd, bcd, maturity, coupon, yieldm)

########################################

metricsDF['RPBP_bps'] = ((metricsDF['MDFunc']*0.0001) - (0.5*metricsDF['ConvexityFunc']*(0.0001**2)))*metricsDF['AllInPrice']
metricsDF['B/E'] = (metricsDF['Yield'] - benchmark)/metricsDF['RPBP_bps']
metricsDF['Sharpe'] = (metricsDF['Yield'] - benchmark)/(metricsDF['RPBP_bps']*100)

#metricsDF['-100bps_CapRt'] = ((metricsDF['MDFunc']*0.01) - (0.5*metricsDF['ConvexityFunc']*(0.01**2)))*metricsDF['AllInPrice']
#metricsDF['-50bps_CapRt'] = ((metricsDF['MDFunc']*0.005) - (0.5*metricsDF['ConvexityFunc']*(0.005**2)))*metricsDF['AllInPrice']
#metricsDF['-25bps_CapRt'] = ((metricsDF['MDFunc']*0.0025) - (0.5*metricsDF['ConvexityFunc']*(0.0025**2)))*metricsDF['AllInPrice']
#metricsDF['0bps_CapRt'] = ((metricsDF['MDFunc']*0) - (0.5*metricsDF['ConvexityFunc']*(0**2)))*metricsDF['AllInPrice']
#metricsDF['+25bps_CapRt'] = ((metricsDF['MDFunc']*0.0025) - (0.5*metricsDF['ConvexityFunc']*(0.0025**2)))*metricsDF['AllInPrice']*-1
#metricsDF['+50bps_CapRt'] = ((metricsDF['MDFunc']*0.005) - (0.5*metricsDF['ConvexityFunc']*(0.005**2)))*metricsDF['AllInPrice']*-1
#metricsDF['+100bps_CapRt'] = ((metricsDF['MDFunc']*0.01) - (0.5*metricsDF['ConvexityFunc']*(0.01**2)))*metricsDF['AllInPrice']*-1

metricsDF['-100bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']-0.01)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
metricsDF['-50bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']-0.005)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
metricsDF['-25bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']-0.0025)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
metricsDF['0bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
metricsDF['+25bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']+0.0025)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
metricsDF['+50bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']+0.005)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
metricsDF['+100bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']+0.01)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)



#metricsDF['AllInPrice_Base'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])))


#metricsDF['-100bps_TotRt'] = ((metricsDF['MDFunc']*0.01) - (0.5*metricsDF['ConvexityFunc']*(0.01**2)))*metricsDF['AllInPrice'] + metricsDF['Yield']
#metricsDF['-50bps_TotRt'] = ((metricsDF['MDFunc']*0.005) - (0.5*metricsDF['ConvexityFunc']*(0.005**2)))*metricsDF['AllInPrice'] + metricsDF['Yield']
#metricsDF['-25bps_TotRt'] = ((metricsDF['MDFunc']*0.0025) - (0.5*metricsDF['ConvexityFunc']*(0.0025**2)))*metricsDF['AllInPrice'] + metricsDF['Yield']
#metricsDF['0bps_TotRt'] = ((metricsDF['MDFunc']*0) - (0.5*metricsDF['ConvexityFunc']*(0**2)))*metricsDF['AllInPrice'] + metricsDF['Yield']
#metricsDF['+25bps_TotRt'] = ((metricsDF['MDFunc']*0.0025) - (0.5*metricsDF['ConvexityFunc']*(0.0025**2)))*metricsDF['AllInPrice']*-1 + metricsDF['Yield']
#metricsDF['+50bps_TotRt'] = ((metricsDF['MDFunc']*0.005) - (0.5*metricsDF['ConvexityFunc']*(0.005**2)))*metricsDF['AllInPrice']*-1 + metricsDF['Yield']
#metricsDF['+100bps_TotRt'] = ((metricsDF['MDFunc']*0.01) - (0.5*metricsDF['ConvexityFunc']*(0.01**2)))*metricsDF['AllInPrice']*-1 + metricsDF['Yield']

metricsDF['-100bps_TotRt'] = metricsDF['-100bps_CapRt'] + metricsDF['Yield']
metricsDF['-50bps_TotRt'] = metricsDF['-50bps_CapRt']  + metricsDF['Yield']
metricsDF['-25bps_TotRt'] = metricsDF['-25bps_CapRt'] + metricsDF['Yield']
metricsDF['0bps_TotRt'] = metricsDF['0bps_CapRt'] + metricsDF['Yield']
metricsDF['+25bps_TotRt'] = metricsDF['+25bps_CapRt'] + metricsDF['Yield']
metricsDF['+50bps_TotRt'] = metricsDF['+50bps_CapRt'] + metricsDF['Yield']
metricsDF['+100bps_TotRt'] = metricsDF['+100bps_CapRt'] + metricsDF['Yield']




@st.cache_data
def metrics(data):

    palette = px.colors.qualitative.Set3

    colors = ['white', palette[1], 'white', palette[1], 'white', palette[1]]

    headerColor = palette[11]
  

    head = ['<b>Term<b>', '<b>Proxy<b>', '<b>Maturity<b>', '<b>Yield<b><b>', '<b>AIP<b>', '<b>Clean<b>', '<b>Acc Int<b>', '<b>Duration<b>', '<b>MD<b>', '<b>Convexity<b>', '<b>Delta<b>', '<b>RPBP (Amt)<b>',
            '<b>RPBP (%)<b>', '<b>Break-Even<b>', '<b>Sharpe<b>']
    

    #data['MD'] = data['MD']*100
    #data['Convexity'] = data['Convexity']*100
    data['AllInPriceVal'] = data['AllInPrice']*100
    data['CleanRoundVal'] = data['CleanRound']*100
    data['AccIntRoundVal'] = data['AccIntRound']*100

    term = data.index.to_list()
    proxy = data['Proxy'].to_list()
    maturity = data['Maturity'].to_list()
    aip = data['AllInPriceVal'].map('{:,.5f}'.format).to_list()
    clean = data['CleanRoundVal'].map('{:,.5f}'.format).to_list()
    acc = data['AccIntRoundVal'].map('{:,.5f}'.format).to_list()
    yield1 = data['Yield'].map('{:.3%}'.format).to_list()
    dur = data['DurationFunc'].map('{:,.3f}'.format).to_list()
    md = data['MDFunc'].map('{:,.3f}'.format).to_list()
    convexity = data['ConvexityFunc'].map('{:,.3f}'.format).to_list()
    delta = data['DeltaFunc'].map('{:,.3f}'.format).to_list()
    rpbp_amt = data['RPBPFunc'].map('{:,.3f}'.format).to_list()
    rpbp_bps = data['RPBP_bps'].map('{:.3%}'.format).to_list()
    be = data['B/E'].map('{:,.0f}'.format).to_list()
    sharpe = data['Sharpe'].map('{:,.3f}'.format).to_list()
    
    
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5,6,7,8,9,10,11,12,13, 14, 15],
        columnwidth = [30,30,30,30,30,30,30,30,30,30,30,30,30, 30, 30],
        
        header=dict(values=head,
                    fill_color=headerColor,
                    line_color='darkslategray',
                    font=dict(color='black'),
                    align=['left', 'center']),
        cells=dict(values=[term, proxy, maturity, yield1, aip, clean, acc, dur, md, convexity, delta, rpbp_amt, rpbp_bps, be, sharpe],
                   fill_color=[colors*15],
                   line_color='darkslategray',
                   font=dict(color='black'),
                   align=['left', 'center']))
    ])   

    fig.update_layout(height=155, width=1425, margin=dict(l=2, r=6, b=2,t=2))
    
    return fig

bmetrics = metrics(metricsDF)
st.plotly_chart(bmetrics)

#st.markdown("<i style='text-align: left; color: charcoal; padding-left: 0px; font-size: 15px'><b>JSE Bond Pricer Link - https://bondcalculator.jse.co.za/BondSingle.aspx?calc=Spot<b></i>", unsafe_allow_html=True)


##########################################################################################################################################################


@st.cache_data
def capitalSensitivity(data):

    palette = px.colors.qualitative.Set3

    colors = ['white', palette[1], 'white', palette[1], 'white', palette[1]]

    headerColor = palette[11]
     
    head = ['<b>Term<b>','<b>-100bps<b>', '<b>-50bps<b>', '<b>-25bps<b>', '<b>0bps<b>', '<b>+25bps<b>', '<b>+50bps<b>', '<b>+100bps<b>' ]

    term = data.index.to_list()
    hundredDown = data['-100bps_CapRt'].map('{:.3%}'.format).to_list()
    fiftyDown = data['-50bps_CapRt'].map('{:.3%}'.format).to_list()
    quarterDown = data['-25bps_CapRt'].map('{:.3%}'.format).to_list()
    base = data['0bps_CapRt'].map('{:.3%}'.format).to_list()
    quarterUp = data['+25bps_CapRt'].map('{:.3%}'.format).to_list()
    fiftyUp = data['+50bps_CapRt'].map('{:.3%}'.format).to_list()
    hundredUp = data['+100bps_CapRt'].map('{:.3%}'.format).to_list()
  
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5,6,7,8],
        columnwidth = [20,30,30,30,30,30,30,30],
        
        header=dict(values=head,
                    fill_color=headerColor,
                    line_color='darkslategray',
                    font=dict(color='black'),
                    align=['left', 'center']),
        cells=dict(values=[term, hundredDown, fiftyDown, quarterDown, base, quarterUp, fiftyUp,hundredUp],
                   fill_color=[colors*8],
                   line_color='darkslategray',
                   font=dict(color='black'),
                   align=['left', 'center']))
    ])   

    fig.update_layout(height=155, width=700, margin=dict(l=2, r=20, b=2,t=2))
    
    return fig


#bmetrics = metrics(metricsDF)
capSensitivity = capitalSensitivity(metricsDF)


#col1, col2 = st.columns([1,1])
#col1.plotly_chart(bmetrics)
#col2.plotly_chart(capSensitivity)

#st.plotly_chart(bmetrics)
#st.plotly_chart(capSensitivity)

#st.dataframe(metricsDF)
###########################################################################################################################################################

@st.cache_data
def totalSensitivity(data):

    palette = px.colors.qualitative.Set3

    colors = ['white', palette[1], 'white', palette[1], 'white', palette[1]]

    headerColor = palette[11]

    head = ['<b>Term<b>','<b>-100bps<b>', '<b>-50bps<b>', '<b>-25bps<b>', '<b>0bps<b>', '<b>+25bps<b>', '<b>+50bps<b>', '<b>+100bps<b>' ]

    term = data.index.to_list()
    hundredDown = data['-100bps_TotRt'].map('{:.3%}'.format).to_list()
    fiftyDown = data['-50bps_TotRt'].map('{:.3%}'.format).to_list()
    quarterDown = data['-25bps_TotRt'].map('{:.3%}'.format).to_list()
    base = data['0bps_TotRt'].map('{:.3%}'.format).to_list()
    quarterUp = data['+25bps_TotRt'].map('{:.3%}'.format).to_list()
    fiftyUp = data['+50bps_TotRt'].map('{:.3%}'.format).to_list()
    hundredUp = data['+100bps_TotRt'].map('{:.3%}'.format).to_list()
  
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5,6,7,8],
        columnwidth = [20,30,30,30,30,30,30,30],
        
        header=dict(values=head,
                    fill_color=headerColor,
                    line_color='darkslategray',
                    font=dict(color='black'),
                    align=['left', 'center']),
        cells=dict(values=[term, hundredDown, fiftyDown, quarterDown, base, quarterUp, fiftyUp,hundredUp],
                   fill_color=[colors*8],
                   line_color='darkslategray',
                   font=dict(color='black'),
                   align=['left', 'center']))
    ])   

    fig.update_layout(height=155, width=700, margin=dict(l=20, r=0, b=2,t=2))
    
    return fig

totSensitivity = totalSensitivity(metricsDF)
#st.plotly_chart(totSensitivity)


col1, col2 = st.columns([1,1])
col1.markdown("<h1 style='text-align: left; color: darkred; padding-left: 0px; font-size: 25px'><b>Capital Returns<b></h1>", unsafe_allow_html=True)
col1.plotly_chart(capSensitivity)
col2.markdown("<h1 style='text-align: left; color: darkred; padding-left: 20px; font-size: 25px'><b>Total Returns<b></h1>", unsafe_allow_html=True)
col2.plotly_chart(totSensitivity)

##############################################################################################################################################################
#####Optimizer################################################################################################################################################

######col3, col4 = st.columns([2,6])
#st.markdown(' ')
#st.header('Portfolio Optimizer')


st.markdown("<h1 style='text-align: left; color: charcoal; padding-left: 0px; font-size: 40px'><b>Expected Return Probability<b></h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: left; color: darkred; padding-left: 0px; font-size: 25px'><b>Scenario Probabilities<b></h1>", unsafe_allow_html=True)

colone,coltwo, colthree = st.columns([1,1,1])
base_move = colone.number_input('Base Shift (%)', min_value=-2.00, max_value=2.00, step=0.25, value=0.00)
base_probability = colone.number_input('Base Probability (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.33)

worst_move = coltwo.number_input('Worst Shift (%)', min_value=-2.00, max_value=2.00, step=0.25, value=0.50)
worst_probability = coltwo.number_input('Worst Probability (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.33)

best_move = colthree.number_input('Best Shift (%)', min_value=-2.00, max_value=2.00, step=0.25, value=-0.50)
best_probability = colthree.number_input('Best Probability (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.34)

metricsDF['BaseMove'] = metricsDF.apply(lambda x: (allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']+(base_move/100))) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])))+ x['Yield'], axis=1)
metricsDF['WorstMove'] = metricsDF.apply(lambda x: (allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']+(worst_move/100))) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])))+ x['Yield'], axis=1)
metricsDF['BestMove'] = metricsDF.apply(lambda x: (allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']+(best_move/100))) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])))+ x['Yield'], axis=1)

metricsDF['BaseMoveProb'] = metricsDF['BaseMove'] * base_probability
metricsDF['WorstMoveProb'] = metricsDF['WorstMove'] * worst_probability
metricsDF['BestMoveProb'] = metricsDF['BestMove'] * best_probability

metricsDF['TotMoveProb'] = metricsDF['BaseMoveProb'] + metricsDF['WorstMoveProb'] + metricsDF['BestMoveProb']


st.markdown("<h1 style='text-align: left; color: darkred; padding-left: 0px; font-size: 25px'><b>Bond Allocation<b></h1>", unsafe_allow_html=True)

a,b,c,d,e,f = st.columns([1,1,1,1,1,1])

rtbAlloc = a.number_input('1Y Alloc (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.15)
r186Alloc = b.number_input('4Y Alloc (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.15)
r2030Alloc = c.number_input('7Y Alloc (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.2)
r2035Alloc = d.number_input('12Y Alloc (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.15)
r2044Alloc = e.number_input('20Y Alloc (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.2)
r2053Alloc = f.number_input('30Y Alloc (%)', min_value=-0.00, max_value=1.00, step=.05, value=0.15)

userAlloc = [rtbAlloc, r186Alloc, r2030Alloc, r2035Alloc, r2044Alloc, r2053Alloc]


metricsDF['uAlloc'] = userAlloc

totProb = metricsDF['uAlloc'].sum()

ScenarioProb = base_probability + worst_probability + best_probability


@st.cache_data
def userAllocation(data, baseProb, WorstProb, BestProb):


    basePro = str('{:.0%}'.format(baseProb))
    worstPro = str('{:.0%}'.format(WorstProb))
    bestPro = str('{:.0%}'.format(BestProb))


    palette = px.colors.qualitative.Set3

    colors = ['white', palette[1], 'white', palette[1], 'white', palette[1]]

    headerColor = palette[11]
     
    head = ['<b>Term<b>','<b>Proxy<b>','<b>Base ('+basePro+')<b>', '<b>Worst ('+worstPro+')<b>', '<b>Best ('+bestPro+')<b>', '<b>User Alloc<b>']

    term = data.index.to_list()
    proxy = data['Proxy'].to_list()
    trBase = data['BaseMove'].map('{:.3%}'.format).to_list()
    trWorst = data['WorstMove'].map('{:.3%}'.format).to_list()
    trBest = data['BestMove'].map('{:.3%}'.format).to_list()
    usAlloc = data['uAlloc'].map('{:.0%}'.format).to_list()

  
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5,6],
        columnwidth = [20,30,30,30,30,30],
        
        header=dict(values=head,
                    fill_color=headerColor,
                    line_color='darkslategray',
                    font=dict(color='black'),
                    align=['left', 'center']),
        cells=dict(values=[term, proxy, trBase, trWorst, trBest, usAlloc],
                   fill_color=[colors*5],
                   line_color='darkslategray',
                   font=dict(color='black'),
                   align=['left', 'center']))
    ])   

    fig.update_layout(height=155, width=1420, margin=dict(l=2, r=2, b=2,t=2))
    
    return fig



st.markdown("<h1 style='text-align: left; color: darkred; padding-left: 0px; font-size: 25px'><b>Results Summary<b></h1>", unsafe_allow_html=True)


useAlloc = userAllocation(metricsDF, base_probability, worst_probability, best_probability)

b1,b2 = st.columns([1,1])


st.plotly_chart(useAlloc)

b1,b2,b3 = st.columns([1,1,1])

if (ScenarioProb == 1):
    b1.success('Sum of Scenario Probabilities = '+str('{:.0%}'.format(ScenarioProb)), icon="✅")
elif (ScenarioProb > 1):
    b1.error('Sum of Scenario Probabilities = '+str('{:.0%}'.format(ScenarioProb)), icon="🚨")
elif (ScenarioProb < 1):
    b1.warning('Sum of Scenario Probabilities = '+str('{:.0%}'.format(ScenarioProb)), icon="⚠️")


if (totProb == 1):
    b2.success('Sum of Bond Allocation = '+str('{:.0%}'.format(totProb)), icon="✅")
elif (totProb > 1):
    b2.error('Sum of Bond Allocation = '+str('{:.0%}'.format(totProb)), icon="🚨")
elif (totProb < 1):
    b2.warning('Sum of Bond Allocation = '+str('{:.0%}'.format(totProb)), icon="⚠️")



metricsDF['ExpRt'] =  metricsDF['TotMoveProb'] * metricsDF['uAlloc']

totPfRet = metricsDF['ExpRt'].sum()

b3.info('Expected Portfolio Return = '+str('{:.3%}'.format(totPfRet)), icon="🚀")
#st.markdown(totProb)
#st.markdown(totPfRet)

#tg_ret=0.08

# Initialize the variables
#a1, a2, a3, a4, a5 = cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True)

# Setup objective
#objective = cp.Maximize(metricsDF['Sharpe_Base'][0]*a1 + metricsDF['Sharpe_Base'][1]*a2 + metricsDF['Sharpe_Base'][2]*a3 + metricsDF['Sharpe_Base'][3]*a4 + metricsDF['Sharpe_Base'][4]*a5)


#R186Alloc = metricsDF['ExpRt'][0]
#R2030Alloc = metricsDF['ExpRt'][1]
#R2035Alloc = metricsDF['ExpRt'][2]
#R2044Alloc = metricsDF['ExpRt'][3]
#R2053Alloc = metricsDF['ExpRt'][4]


# Initialize constraints (must be a list)
#constraints = [(R186Alloc)*a1 + (R2030Alloc)*a2 + (R2035Alloc)*a3 + (R2044Alloc)*a4 + (R2053Alloc)*a5 == tg_ret, a1+a2+a3+a4+a5 == 1, a1<=1, a2<=1, a3<=1, a4<=1, a5<=1]

# Create our problem
#problem = cp.Problem(objective, constraints)

# Solve our problem
#problem.solve()


#st.markdown(np.round(a1.value,2))
#st.markdown(np.round(a2.value,2))
#st.markdown(np.round(a3.value,2))
#st.markdown(np.round(a4.value,2))
#st.markdown(np.round(a5.value,2))

###########################################################################################################################
#def allocUser(data):
#
#    #counter1 = len(data.index)
#    #counter0 = list(range(1,counter1+1))
#
#    figBucket = px.bar(data, x=data.index, y='uAlloc',
#                hover_data=['Proxy', 'Alloc_Base'], 
#                color='Alloc_Base',
#                #color_continuous_scale=px.colors.sequential.RdBu_r, 
#                text_auto='.2%',
#                labels={'uAlloc':'Alloc', 'Period': 'Term'}, height=500, width=680)
#
#
#    figBucket.update_layout(
#        #xaxis = dict(
#        #    tickmode = 'array',
#        #    tickvals = counter0,
#        #    ticktext = data.index.to_list()
#        #),
#        title=dict(text='Allocation',
#                font=dict(size=30),
#                automargin=True,
#                yref='paper'),
#        margin=dict(l=0, r=0, b=0,t=50)
#    )
#
#    figBucket.update_traces(textfont_size=12,
#                            textangle=0,
#                            textposition="outside",
#                            cliponaxis=False,
#                            marker_line_color='rgb(8,48,107)',
#                            marker_line_width=1.5,
#                            opacity=1
#                            )
#
#
#    return figBucket
#
#allocationUserGraph = allocUser(metricsDF)
#st.plotly_chart(allocationUserGraph)
###########################################################################################################

#st.dataframe(metricsDF)


#metricsDF['-100bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']-0.01)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
#metricsDF['-50bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']-0.005)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)
#metricsDF['-25bps_CapRt'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield']-0.0025)) - allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'])), axis=1)



st.markdown(" ")
st.markdown(" ")
st.markdown("<h1 style='text-align: left; color: charcoal; padding-left: 0px; font-size: 40px'><b>Portfolio Optimizer<b></h1>", unsafe_allow_html=True)


#base_shift = st.sidebar.number_input('Overall  Curve Shift (%)', min_value=-5.00, max_value=5.00, format='%0.3f', step=0.25, value=0.00)
st.sidebar.markdown(' ')
#st.sidebar.markdown('Curve Change')
st.sidebar.markdown("<h1 style='text-align: left; color: #008080; padding-left: 0px; font-size: 20px'><b>Curve Change<b></h1>", unsafe_allow_html=True)
base_shift = st.sidebar.number_input('Overall Shift (%)', min_value=-2.00, max_value=2.00, format='%0.3f', step=0.25, value=0.00)

col3, colx, col4 = st.columns([1,0.1, 1])

col3.markdown(" ")
col3.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
opt = col3.radio("Optimization Style",("Maximum Sharpe", "Target Return", "Target Risk"))

if (opt=='Target Return'):
    barHeight=315
    target_input = col3.number_input('Total Return (%)', min_value=0.00, max_value=20.00, format='%0.3f', step=0.25, value=11.00)
    target=target_input/100
elif(opt=='Target Risk'):
    barHeight=315
    target_rpbp = col3.number_input('Total RPBP (%)', min_value=0.00, max_value=1.00, format='%0.3f', step=0.01, value=0.04)
    Trpbp = target_rpbp/100  
else:
    barHeight=315
    


st.markdown(' ')
#st.markdown('Curve Shift')
############base_shift = st.number_input('Overall  Curve Shift (%)', min_value=-5.00, max_value=5.00, format='%0.3f', step=0.25, value=0.00)
#base_prob = st.number_input('Base Probability', min_value=0.00, max_value=1.00, format='%0.3f', step=0.25, value=1.00)
metricsDF['AllInPrice_Base'] = metricsDF.apply(lambda x: allInPrice(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'] + base_shift/100)), axis=1)
metricsDF['CapPrice_Base'] = metricsDF['AllInPrice_Base'] - metricsDF['AllInPrice']
#metricsDF['TotPrice_Base'] = metricsDF['CapPrice_Base'] + (metricsDF['Yield'] + base_shift/100)
metricsDF['TotPrice_Base'] = metricsDF['CapPrice_Base'] + (metricsDF['Yield'])

metricsDF['MDFunc_Base'] = metricsDF.apply(lambda x: modfinal(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'] + base_shift/100)), axis=1)
metricsDF['ConvexityFunc_Base'] = metricsDF.apply(lambda x: convexityfinal(settleDate, x['LCD'], x['NCD'], x['BCD'], x['Maturity'], x['Coupon'], (x['Yield'] + base_shift/100)), axis=1)
metricsDF['RPBP_bps_Base'] = ((metricsDF['MDFunc_Base']*0.0001) - (0.5*metricsDF['ConvexityFunc_Base']*(0.0001**2)))*metricsDF['AllInPrice_Base']
#metricsDF['Sharpe_Base'] = ((metricsDF['Yield']+(base_shift/100)) - benchmark)/(metricsDF['RPBP_bps_Base']*100)
#metricsDF['Sharpe_Base'] = ((metricsDF['Yield']) - benchmark)/(metricsDF['RPBP_bps_Base']*100)
metricsDF['Sharpe_Base'] = ((metricsDF['TotPrice_Base']) - benchmark)/(metricsDF['RPBP_bps_Base']*100)


RtbYieldBase = (metricsDF['TotPrice_Base'][0]*100)
R186YieldBase = (metricsDF['TotPrice_Base'][1]*100)
R2030YieldBase = (metricsDF['TotPrice_Base'][2]*100)
R2035YieldBase = (metricsDF['TotPrice_Base'][3]*100)
R2044YieldBase = (metricsDF['TotPrice_Base'][4]*100)
R2053YieldBase = (metricsDF['TotPrice_Base'][5]*100)


RtbRiskBase = (metricsDF['RPBP_bps_Base'][0]*100)
R186RiskBase = (metricsDF['RPBP_bps_Base'][1]*100)
R2030RiskBase = (metricsDF['RPBP_bps_Base'][2]*100)
R2035RiskBase = (metricsDF['RPBP_bps_Base'][3]*100)
R2044RiskBase = (metricsDF['RPBP_bps_Base'][4]*100)
R2053RiskBase = (metricsDF['RPBP_bps_Base'][5]*100)



################################################################################################################################
# Initialize the variables
x1, x2, x3, x4, x5, x6 = cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True)
objective = cp.Maximize(metricsDF['Sharpe_Base'][0]*x1 + metricsDF['Sharpe_Base'][1]*x2 + metricsDF['Sharpe_Base'][2]*x3 + metricsDF['Sharpe_Base'][3]*x4 + metricsDF['Sharpe_Base'][4]*x5 + metricsDF['Sharpe_Base'][5]*x6)


# Initialize constraints (must be a list)
if(opt == 'Target Return'):
    constraints = [(RtbYieldBase)*x1 + (R186YieldBase)*x2 + (R2030YieldBase)*x3 + (R2035YieldBase)*x4 + (R2044YieldBase)*x5 + (R2053YieldBase)*x6 == target_input, x1+x2+x3+x4+x5+x6 == 1, x1<=1, x2<=1, x3<=1, x4<=1, x5<=1, x6<=1]
elif(opt == 'Target Risk'):
    constraints = [(RtbRiskBase)*x1 + (R186RiskBase)*x2 + (R2030RiskBase)*x3 + (R2035RiskBase)*x4 + (R2044RiskBase)*x5 + (R2053RiskBase)*x6 == target_rpbp, x1+x2+x3+x4+x5+x6 == 1, x1<=1, x2<=1, x3<=1, x4<=1, x5<=1, x6<=1]
else:
    constraints = [x1+x2+x3+x4+x5+x6 == 1, x1<=1, x2<=1, x3<=1, x4<=1, x5<=1, x6<=1]
#################################################################################################################################


#constraints = [metricsDF['TotPrice_Base'][0]*x1 + metricsDF['TotPrice_Base'][1]*x2 + metricsDF['TotPrice_Base'][2]*x3 + metricsDF['TotPrice_Base'][3]*x4 + metricsDF['TotPrice_Base'][4]*x5 == target_input, x1+x2+x3+x4+x5 == 1, x1<=1, x2<=1, x3<=1, x4<=1, x5<=1]
# Create our problem
problem = cp.Problem(objective, constraints)

# Solve our problem
problem.solve()

#st.markdown(np.round(x1.value,2))
#st.markdown(np.round(x2.value,2))
#st.markdown(np.round(x3.value,2))
#st.markdown(np.round(x4.value,2))
#st.markdown(np.round(x5.value,2))

###st.dataframe(metricsDF)

#base_weights=[np.round(x1.value,2),np.round(x2.value,2),np.round(x3.value,2),np.round(x4.value,2),np.round(x5.value,2)]
base_weights=[x1.value, x2.value, x3.value, x4.value, x5.value, x6.value]
metricsDF['Alloc_Base'] = base_weights

@st.cache_data
def baseAllocation(data):

    palette = px.colors.qualitative.Set3

    colors = ['white', palette[1], 'white', palette[1], 'white', palette[1]]

    headerColor = palette[11]
     
    head = ['<b>Term<b>','<b>Proxy<b>','<b>Total Return<b>', '<b>MD<b>', '<b>Conv<b>','<b>RPBP (Risk)<b>','<b>Sharpe<b>', '<b>Allocation<b>']

    term = data.index.to_list()
    proxy = data['Proxy'].to_list()
    trBase = data['TotPrice_Base'].map('{:.3%}'.format).to_list()
    mdBase = data['MDFunc_Base'].map('{:,.3f}'.format).to_list()
    convBase = data['ConvexityFunc_Base'].map('{:,.3f}'.format).to_list()
    rpbpBase = data['RPBP_bps_Base'].map('{:.3%}'.format).to_list()

    sharpeBase = data['Sharpe_Base'].map('{:,.3f}'.format).to_list()
    allocBase = data['Alloc_Base'].map('{:.0%}'.format).to_list()
  
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5,6,7,8],
        columnwidth = [20,30,30,30,30,30,30, 30],
        
        header=dict(values=head,
                    fill_color=headerColor,
                    line_color='darkslategray',
                    font=dict(color='black'),
                    align=['left', 'center']),
        cells=dict(values=[term, proxy, trBase, mdBase, convBase, rpbpBase, sharpeBase, allocBase],
                   fill_color=[colors*8],
                   line_color='darkslategray',
                   font=dict(color='black'),
                   align=['left', 'center']))
    ])   

    fig.update_layout(height=155, width=700, margin=dict(l=2, r=20, b=2,t=2))
    
    return fig

baseAlloc = baseAllocation(metricsDF)


col3.markdown("<h1 style='text-align: left; color: darkred; padding-left: 0px; font-size: 25px'><b>Portfolio Metrics<b></h1>", unsafe_allow_html=True)
#st.plotly_chart(baseAlloc)




#st.markdown(np.round(x1.value,2))
#st.markdown(np.round(x2.value,2))
#st.markdown(np.round(x3.value,2))
#st.markdown(np.round(x4.value,2))
#st.markdown(np.round(x5.value,2))









#########################################
#
## Initialize the variables
#x1, x2, x3, x4, x5 = cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True)
#objective = cp.Minimize(metricsDF['RPBP_bps'][0]*x1 + metricsDF['RPBP_bps'][1]*x2 + metricsDF['RPBP_bps'][2]*x3 + metricsDF['RPBP_bps'][3]*x4 + metricsDF['RPBP_bps'][4]*x5)
#
## Initialize constraints (must be a list)
#constraints = [R186YieldAdj*x1 + R2030YieldAdj*x2 + R2035YieldAdj*x3 + R2044YieldAdj*x4 + R2053YieldAdj*x5 == target_input, x1+x2+x3+x4+x5 == 1, x1<=1, x2<=1, x3<=1, x4<=1, x5<=1]
#
## Create our problem
#problem = cp.Problem(objective, constraints)
#
## Solve our problem
#problem.solve()
#
#st.markdown(np.round(x1.value,2))
#st.markdown(np.round(x2.value,2))
#st.markdown(np.round(x3.value,2))
#st.markdown(np.round(x4.value,2))
#st.markdown(np.round(x5.value,2))
#
######################################

#st.dataframe(metricsDF)

def allocProfile(data, col, mtype, txt, height=315):

    #counter1 = len(data.index)
    #counter0 = list(range(1,counter1+1))

    figBucket = px.bar(data, x=data.index, y=col,
                hover_data=['Proxy', col], 
                color=col,
                #color_continuous_scale=px.colors.sequential.RdBu_r, 
                text_auto=txt,
                labels={col:mtype, 'Period': 'Term'}, height=height, width=680)


    figBucket.update_layout(
        #xaxis = dict(
        #    tickmode = 'array',
        #    tickvals = counter0,
        #    ticktext = data.index.to_list()
        #),
        title=dict(text=mtype,
                font=dict(size=30),
                #automargin=True,
                yref='paper'),
                margin=dict(l=0, r=0, b=0,t=50)
    )

    figBucket.update_traces(textfont_size=12,
                            textangle=0,
                            textposition="outside",
                            cliponaxis=False,
                            marker_line_color='rgb(8,48,107)',
                            marker_line_width=1.5,
                            opacity=1
                            )


    return figBucket

#allocationGraph = allocProfile(metricsDF, barHeight)

#st.dataframe(metricsDF)

#col3, col4 = st.columns([1,1])

col3.plotly_chart(baseAlloc)

col4.markdown("<h1 style='text-align: left; color: darkred; padding-left: 0px; font-size: 25px'><b> <b></h1>", unsafe_allow_html=True)

choices = ['Allocation', 'Convexity', 'MD', 'RPBP', 'Sharpe', 'Total Return']

metric = col4.selectbox("Metric", choices)

if(metric == 'Allocation'):
    styleType='Allocation'
    styleCol = 'Alloc_Base'
    styleText='.0%'
elif(metric == 'Convexity'):
    styleType='Convexity'
    styleCol = 'ConvexityFunc_Base'
    styleText='.2f'
elif(metric == 'MD'):
    styleType='MD'
    styleText='.2f'
    styleCol = 'MDFunc_Base'
elif(metric == 'RPBP'):
    styleType='RPBP'
    styleCol = 'RPBP_bps_Base'
    styleText='.3%'
elif(metric == 'Sharpe'):
    styleType='Sharpe'
    styleCol='Sharpe_Base'
    styleText='.2f'
else:
    styleType='Total Return'
    styleCol='TotPrice_Base'
    styleText='.3%'

allocationGraph = allocProfile(metricsDF, styleCol,  styleType, styleText, barHeight)
col4.plotly_chart(allocationGraph)

st.markdown("<i style='text-align: left; color: charcoal; padding-left: 0px; font-size: 15px'><b>JSE Bond Pricer Link - https://bondcalculator.jse.co.za/BondSingle.aspx?calc=Spot<b></i>", unsafe_allow_html=True)


