#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 03:22:16 2017

@author: geyi0530
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 02:26:05 2017

@author: geyi0530
"""

# encoding: UTF-8
import json
from os.path import join,dirname
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud import PersonalityInsightsV3
import json

DIRECTORY='debbie.mp3'
DIRECTORY2='profile.json'

##used to transform speech to text
#argument
#    directory -- path to file
#result
#   string -- json format answer, need further operation
def spe2tex(directory):
    speech_to_text = SpeechToTextV1(
                                    username='7a20510f-9402-456f-bc13-30caf57a2b56',
                                    password='iqRBHZjWIouq',
                                    x_watson_learning_opt_out=False
                                    )
    with open(directory,'rb')as audio_file:
        return (json.dumps(speech_to_text.recognize(
                                                   audio_file,content_type='audio/mp3',timestamps=True,
                                                   model='en-US_BroadbandModel',word_confidence=True
                                                   ),indent=2,encoding='UTF-8',ensure_ascii=False))
        
        
def res2string(res):
    sss=json.loads(res)
    final=''
    total=0
    count=0
    group=[]
    confG=[]
    for result in sss['results']:
        sentence=result['alternatives'][0]['transcript']
        confidence=result['alternatives'][0]['confidence']
        print(sentence)
        print(confidence)
        final+=sentence
        total+=confidence
        count+=1
        group.append(sentence)
        confG.append(confidence)
    avg=total/count
    print final,avg
    f=open("sentence.txt", "w")
    f.write(final)      
    return final  #only return to that string (one passage)

def string2cha(text):
    text+='''
    ,
    content_type='text/plain',
    content_language=None,
    accept='application/json',
    accept_language=None,
    raw_scores=False,
    consumption_preferences=False,
    csv_headers=False
    '''
    f=open('profile.json','w')
    f.write(text)
    f.close()

def cha2result(directory2):
    personality_insights = PersonalityInsightsV3(
                                                 version='2016-10-20',
                                                 username='39b8b2c3-4c89-4744-9b13-5a2c819e336a',
                                                 password='aqX0fmoCPtMA'
                                                 )  
    
    with open(directory2) as profile_json:
         profile = personality_insights.profile(
                                                profile_json.read(), content_type='text/plain',
                                                raw_scores=True, consumption_preferences=True)
    character_text=json.dumps(profile, indent=2)
#print(character_text)

    sss=json.loads(character_text)
    final='Personality:'
    final
#print ("Personality:")
    for result in sss['values']:
        character1=result['name']
        percent1=result['percentile']
        final+="\n"+character1
        final+="\n"+str(percent1)

    final+="\n\n"+"Value:"
#print ("\n"+"Value:")
    for result in sss['personality']:
        character2=result['name']
        percent2=result['percentile']
        final+="\n"+character2
        final+="\n"+str(percent2)
    #print character2 
    #print percent2
    print(final)
    return final

if __name__ == "__main__":
    res=spe2tex(DIRECTORY)
    text=res2string(res)
    string2cha(text)  #update jason
    final_string=cha2result(DIRECTORY2)