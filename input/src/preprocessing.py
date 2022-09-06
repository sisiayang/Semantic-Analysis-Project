from opencc import OpenCC
from tqdm import tqdm

def s2twp(messages):
    cc = OpenCC('s2twp')  # s2twp為簡體轉台灣繁體(包含慣用語)
    # 簡轉繁
    changed_messages = []
    for i in tqdm(messages):
        changed_messages.append(cc.convert(i))
    
    return changed_messages

# 回傳只包含中文字符的message
def only_contains_chinese(messages):
    punctuation = '。.，,?？!！~'
    changed_messages = []
    for mes in tqdm(messages):
        more_than_one = 0
        has_chinese = 0
        new = ''
        if(str(mes) == 'nan'):
            changed_messages.append(str(mes))
            continue
        for i in mes:
            if('\u4e00' <= i <= '\u9fa5'):
                if(more_than_one == 1):
                    new += ''
                
                more_than_one = 0
                new += i
                has_chinese = 1

            elif(i in punctuation):
                # 若是句子中有中文才需留下標點，若沒有則可刪掉
                if(has_chinese == 1):
                    new += i
                    more_than_one = 0
                    has_chinese = 0
            else:
                more_than_one = 1

        # 移除頭尾空白和換行
        new = new.strip()
        changed_messages.append(new)
    return changed_messages