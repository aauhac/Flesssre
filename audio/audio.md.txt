음성 편집

wav 파일을 불러와 실행

``` python 

from pydub import AudioSegment
from pydub.playback import play

audio = AudioSegment.from_file("titanium.wav")
play(audio)

```

전체 샘플의 수와 샘플링 레이트를 구해서 샘플링 레이트와 음성의 길이 계산

``` python
import soundfile as sf
data, sample_rate = sf.read("titanium.wav")

print("샘플링 레이트 :", sample_rate)
print("길이 :", len(data) / sample_rate)

```

샘플링 레이트 : 44100
길이 : 106.2

---

샘플링 레이트를 22kHz인 22000로 바꿔준 후 음성 변환

``` python

sample_r = 22000
audio22 = audio.set_frame_rate(sample_r)
audio22.export("titanium22.wav", format="wav")

```

음성 파형을 그리기 위해 librosa 가져오기. 파형을 시각화 해주는 waveshow를 가져와 matplotlib로 시각화

``` python

import librosa
import matplotlib.pyplot as plt
import librosa.display

y , sr = librosa.load('titanium.wav')
plt.figure(figsize=(16, 6))
librosa.display.waveshow(y=y, sr=sr)
plt.show()
 
```

푸리에 변환과 스펙트로그램을 나타내기 위해 librosa 사용. 
오디오 시계열을 사용해 계산하여 시각화

데시벨로 변환한 값을 계산해 스펙트로그램 시각화

``` python 

D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))


print(D.shape)

plt.figure(figsize=(16,6))
plt.plot(D)
plt.show()

```

```python

DB = librosa.amplitude_to_db(D, ref=np.max)

plt.figure(figsize=(16,6))
librosa.display.specshow(DB,sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

```

pydub을 사용해 가져온 음성을 밀리초로 시간을 구분해 잘라준 후 저장

``` python

from pydub import AudioSegment
                  
audio = AudioSegment.from_file("titanium.wav")

start_time = 10000
end_time = 20000

cut_audio = audio[start_time:end_time]

cut_audio.export("sltita.wav", format="wav")

```