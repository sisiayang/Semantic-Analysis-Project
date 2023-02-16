from get_data import get_ownerFbid, get_messages, get_combine_message
from preprocessing import s2twp, only_contains_chinese
from get_tokenize import get_data_token

if __name__ == '__main__':
    OwnerFbid_df = get_ownerFbid(input_path='../orig_messages/ego_survey.csv', output_path='../orig_messages/OwnerFbid.csv')
    OwnerFbid = OwnerFbid_df['OwnerFbid'].tolist()

    input_list = ['../orig_messages/comments.csv', '../orig_messages/comments_comments.csv', '../orig_messages/allfeed.csv']
    orig_output_list = ['../../output/comments.csv', '../../output/comments_comments.csv', '../../output/allfeed.csv']
    combine_output_list = ['../messages_combine_by_Fbid/comments.csv', '../messages_combine_by_Fbid/comments_comments.csv', '../messages_combine_by_Fbid/allfeed.csv']
    processed_output_list = ['../messages_to_be_analyzed/comments.csv', '../messages_to_be_analyzed/comments_comments.csv', '../messages_to_be_analyzed/allfeed.csv']
    data_tokenize_output_list = ['../messages_tokenize/comments.txt', '../messages_tokenize/comments_comments.txt', '../messages_tokenize/allfeed.txt']
    print('process dataframe...')
    for i in range(3):
        df = get_messages(input_list[i], orig_output_list[i], OwnerFbid)
        combine_df = get_combine_message(df, combine_output_list[i])

        # 先以未合併的資料集做處理
        message = s2twp(df['message'].tolist())
        message = only_contains_chinese(message)
        df['message'] = message
        df.to_csv(processed_output_list[i], index=False)
        print('df size check: ', len(df))

        get_data_token(df, data_tokenize_output_list[i])
        
        print('done\n--------------------------------------------------')

    