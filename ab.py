import pandas as pd
import numpy as np
from scipy.stats import stats, binom,norm,zscore
import seaborn as sns
import math

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def build_dataset(control_size=1000,variation_size=1000,control_cr=.4,variation_cr=.5):
    #create control
    control_results = np.random.rand(control_size)<control_cr
    control_df = pd.DataFrame(control_results.astype(int),columns=['result'])
    control_df['group']='control'

    #create variation
    variation_results = np.random.rand(variation_size)<variation_cr
    variation_df = pd.DataFrame(variation_results.astype(int),columns=['result'])
    variation_df['group']='variation'

    df = pd.concat([control_df,variation_df],axis=0)
    return(df)

def graph_distributions(df):
    ###
    #Plot the distributions of both Control & Variations from a an AB test
    ###

    probs =  df.groupby('group').mean()
    control_p = probs.loc['control']
    variation_p = probs.loc['variation']

    control = binom.rvs(size=1000,n=1000,p=control_p)
    variation = binom.rvs(size=1000,n=1000,p=variation_p)

    plt.Figure()
    sns.distplot(control,label='Control')
    sns.distplot(variation,label='Variation')
    plt.title("Binomial Distributions for Control & Variation")
    plt.legend(loc='upper right')
    #plt.close()
    return

def ci(cr,n,alpha):
    ###
    #Get the Confidence Interval for a given proportion of a given sample size
    ###
    z_alpha = norm.ppf(1-alpha/2)
    ci = z_alpha*math.sqrt((cr*(1-cr)/n))
    return (cr-ci,cr+ci)

def hypothesis_test_prop(cr1,cr2,n,test='one'):
    ###
    # Sample sizes are equivalent
    #TODO ADD two-sided test logic
    ###

    se1 = math.sqrt((cr1*(1-cr1)/n))
    se2 = math.sqrt((cr2*(1-cr2)/n))
    se_diff = math.sqrt(se1**2+se2**2)

    z_score = ((cr2 - cr1) / se_diff)

    z = np.arange(-3,3,0.1)
    plt.plot(z,norm.pdf(z))

    if test!='two' and cr1<cr2:
        plt.fill_between(z[z>z_score],norm.pdf(z[z>z_score]))
        p_value = 1-norm.cdf(z_score)
    elif test!='two' and cr1>cr2:
        plt.fill_between(z[z<z_score],norm.pdf(z[z<z_score]))
        p_value = norm.cdf(z_score)

    plt.title("Hypothesis Test for cr1: %s & cr2: %s & %s samples"%(cr1,cr2,n))
    plt.text(-3,.3,'p-value: {:6.3f}'.format(p_value))
    plt.show()

    return(z_score,p_value)

def get_power(h0,h1,n,alpha=0.05,test='two'):
    # ADD FOR single tail
    z_alpha = norm.ppf(1-alpha)
    se0 = np.sqrt(h0*(1-h0)/n) #standard error for proportion
    se1 = np.sqrt(h1*(1-h1)/n) #standard error for proportion

    upper_bound = h0+se0*z_alpha
    lower_bound = h0-se0*z_alpha

    rv = norm(h0,se0)
    rv2 = norm(h1,se1)


    x = np.linspace(h0 - 5*se0, h0 + 5*se0, 100)
    plt.plot(x, rv.pdf(x))
    plt.plot(x, rv2.pdf(x))
    plt.fill_between(x[x<lower_bound],rv2.pdf(x[x<lower_bound]),color='r',alpha=.2)
    plt.fill_between(x[x>upper_bound],rv2.pdf(x[x>upper_bound]),color='r',alpha=.2)
    plt.axvline(x=lower_bound,color='r')
    plt.axvline(x=upper_bound,color='r')

    plt.show()



    lower_alpha = rv2.cdf(lower_bound)
    upper_alpha = (1-rv2.cdf(upper_bound))

    print(lower_alpha)
    print(upper_alpha)

    if test == 'two':
        power = lower_alpha + upper_alpha
    elif test == 'left':
        power = lower_alpha
    elif test == 'right':
        power = upper_alpha

    return(power,lower_bound,upper_bound)


def power_test(p1,p2,beta=0.8, alpha=0.05):

    kappa=1
    alpha=0.05
    beta=0.20
    nB=(p1*(1-p1)/kappa+p2*(1-p2))* (((norm.ppf(1-alpha/2)+norm.ppf(1-beta))/(p1-p2) ))**2
    return(nB)
    #z=(p1-p2)/sqrt(p1*(1-p1)/nB/kapp1+p2*(1-p2)/nB)
    #(Power=pnorm(z-qnorm(1-alpha/2))+pnorm(-z-qnorm(1-alpha/2)))
