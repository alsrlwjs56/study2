from easyocr import Reader

# user-network


# reader = Reader(['en', 'ko'], gpu=False, user_network_directory='C:/EasyOCR')



# result = reader.readtext('D:/format_pc_intro_015@2x.jpg')

# print(result)


# 세월 따라 높이 나빠진 아내가 당신 글쎄늘 
# 선명해서 일기 줄으니 음보해보다 합나다 
# 아내가 건강하길 바라면 사랑물 담야

# 세월 따라 눈이 나빠진 아내가 당신 글씨는 
# 선명해서 잎기 좋으니 응모해보화 합다다 
# 아내가 전강하길 바라여 사랑을 담아

# 세월 따라 눈이 나빠진 아내가 당신 글씨는
# 선명해서 읽기 좋으니 응모해보라 합니다
# 아내가 건강하길 바라며 사랑을 담아

# print('-' * 50)


# print('세월 따라 눈이 나빠진 아내가 당신 글씨는')
# print('선명해서 일기 줄으니 음보해보다 합나다')
# print('아내가 건강하길 바라면 사랑물 담야')

# print('-' * 50)

# print('세월 따라 눈이 나빠진 아내가 당신 글씨는')
# print('선명해서 잎기 좋으니 응모해보화 합다다')
# print('아내가 전강하길 바라여 사랑을 담아')

# print('-' * 50)

ground_truth = '세월 따라 눈이 나빠진 아내가 당신 글씨는 선명해서 읽기 좋으니 응모해보라 합니다 아내가 건강하길 바라며 사랑을 담아'

our_output = '세월 따라 높이 나빠진 아내가 당신 글쎄늘 선명해서 읽기 줄으니 음보해보다 합나다 아내가 건강하길 바라면 사랑을 담야'

easyocr_output = '세월 따라 눈이 나빠진 아내가 당신 글씨는 선맹해서 잎기 좋으니 응모해보화 합다다 아내가 전강하길 바라여 사랑을 담아'
# easyocr_output = '세월 따라 눈이 나빠진 아내가 당신 글씨는 선명해서 잎기 좋으니 응모해보화 합다다 아내가 전강하길 바라여 사랑을 담아'

# Word based Exactly Matching

def word_based_exactly_matching(ground_truth, our_output, easyocr_output):
    ground_truth = ground_truth.split(' ')
    our_output = our_output.split(' ')
    easyocr_output = easyocr_output.split(' ')

    ground_truth_len = len(ground_truth)
    our_output_len = len(our_output)
    easyocr_output_len = len(easyocr_output)

    our_correct = 0
    easyocr_correct = 0

    for i in range(ground_truth_len):
        if ground_truth[i] == our_output[i]:
            our_correct += 1
        if ground_truth[i] == easyocr_output[i]:
            easyocr_correct += 1

    our_accuracy = our_correct / our_output_len
    easyocr_accuracy = easyocr_correct / easyocr_output_len

    print('-' * 50)
    print('' * 50)
    print('Our score : ', our_accuracy)
    print('' * 50)
    print('-' * 50)
    print('' * 50)
    
    print('EasyOCR score : ', easyocr_accuracy)
    print('' * 50)
    
    print('-' * 50)
    
    
    return our_accuracy, easyocr_accuracy

# (0.5882352941176471, 0.6470588235294118)

print(word_based_exactly_matching(ground_truth, our_output, easyocr_output))



import matplotlib.pyplot as plt

# Define precision and recall scores
Ourscore = 0.5882352941176471
EasyOCRscore = 0.6470588235294118

# Create a bar chart with precision and recall scores
fig, ax = plt.subplots()
ax.bar(['Ourscore', 'EasyOCRscore'], [Ourscore, EasyOCRscore], color=['blue', 'green'])

# Add labels and title
ax.set_ylabel('Score')
ax.set_title('OCR Project Evaluation Index')
ax.set_ylim(0, 1)

# Display the chart
plt.show()






