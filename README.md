# F2M_model_9_11

## F2M_model_V9_2
* Depth wise + point wise (Depth-wise Separable Convolution, Xception model에 사용했던 기법)을 Residual block에 적용(채널에 대한 정보 + spatial space 정보를 각각 분리한것). Dilated rate 한 블록을 거칠때마다 1씩 증가. Encoder 와 decoder 사이에 추가
* 나이에 대한 loss 새롭게 구성 
* Same Age loss=MAX(0,𝑒^(−dis 2.77/100) )  , 감소함수
* Different Age loss=ln⁡〖(MAX(1,dis)+𝜖)〗
* 





