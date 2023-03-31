import os
from pydub import AudioSegment

def cut_song(path: str, name: str, class_num: bool):
    '''
    This function cuts the songs to have 1 minute length so that all songs have the same length

    Input:
    - path: path of the song
    - name: name to save the new short version of the song
    - class_num: 0 for songs I like, 1 for songs I dislike
    '''
    AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"
    sound = AudioSegment.from_mp3(path)
    #Selecting Portion we want to cut
    StrtMin, StrtSec, EndMin, EndSec = 0, 0, 1, 1
    # Time to milliseconds conversion
    StrtTime = StrtMin*60*1000+StrtSec*1000
    EndTime = EndMin*60*1000+EndSec*1000
    # Opening file and extracting portion of it
    extract = sound[StrtTime:EndTime]
    # Saving file in required location
    if(class_num == 0):
        extract.export('data_cut/Like/'+name, format="mp3")
    else:
        extract.export('data_cut/Dislike/'+name, format="mp3")
    # new file portion.mp3 is saved at required location


# Read all songs and cut them into 1 min long
def cut_dataset():
    class_ = ["data/Like/" , "data/Dislike/"]
    class_num = 0
    for cla in class_:
        cont = 1
        for fil in os.listdir(cla):
            print(fil)
            name = str(cont)
            while(len(name)<3):
                name = '0'+name
            cut_song(cla+fil, name+'.mp3', class_num)
            cont += 1
        class_num += 1