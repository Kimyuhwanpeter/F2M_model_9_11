# F2M_model_9_11

## F2M_model_V9_2
* Depth wise + point wise (Depth-wise Separable Convolution, Xception model에 사용했던 기법)을 Residual block에 적용(채널에 대한 정보 + spatial space 정보를 각각 분리한것). Dilated rate 한 블록을 거칠때마다 1씩 증가. Encoder 와 decoder 사이에 추가
* 나이에 대한 loss 새롭게 구성 
* Same Age loss=MAX(0,𝑒^(−dis 2.77/100) )  , 감소함수
* Different Age loss=ln⁡〖(MAX(1,dis)+𝜖)〗
<br/>

## F2M_model_V10
* Depth wise + point wise (Depth-wise Separable Convolution, Xception model에 사용했던 기법)을 Residual block에 적용(채널에 대한 정보 + spatial space 정보를 각각 분리한것). Dilated rate 한 블록을 거칠때마다 2씩 증가. Attention 추가. Encoder 내부에도 추가
* 나이에 대한 loss 새롭게 구성 (F2M_model_V9_2 와 동일)
<br/>

## F2M_model_V11
* Depth wise + point wise (Depth-wise Separable Convolution, Xception model에 사용했던 기법)을 Residual block (Dilated rate 한 블록을 거칠때마다 2씩 증가.)에 적용(채널에 대한 정보 + spatial space 정보를 각각 분리한것). Attention 추가. Decoder에 reverse residual block (Dilated rate 한 블록을 거칠때마다 4씩 증가.) 추가.
* 나이에 대한 loss 새롭게 구성 (F2M_model_V9_2 와 동일)
<br/>

## F2M_model_V11_2
* Depth wise + point wise (Depth-wise Separable Convolution, Xception model에 사용했던 기법)을 Residual block (Dilated rate 한 블록을 거칠때마다 2씩 증가.)에 적용(채널에 대한 정보 + spatial space 정보를 각각 분리한것). Attention 추가. Decoder에 reverse residual block (Dilated rate 한 블록을 거칠때마다 4씩 증가.) 추가. 각 block에 있는 1x1 conv를 1x1 depth wise conv 로 변경. ***1x1 depth conv는 adaptive contrast in image 효과가 있다고 판단됨(아직 정확한 실험은 이루어지지 않고, 내 이론 상)***
* 나이에 대한 loss 새롭게 구성 (F2M_model_V9_2 와 동일)








