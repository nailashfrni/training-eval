import pandas as pd

def get_truncated_sentence(sentence, percentage=100):
  '''
  sentece (str): sentence to be truncated
  percentage (float/int): how many percent the sentence would be returned
  '''
  words = sentence.split()
  num_to_remove = int(len(words) * (percentage / 100))
  reduced_sentence = words[num_to_remove:]
  return " ".join(reduced_sentence)

def main(num_percentage=50):
    ...
    df = pd.read_csv('data/mmlu_dataset.csv')
    df['Question'] = df['Question'].apply(lambda x: get_truncated_sentence(x, num_percentage))
    df.to_csv(f'data/mmlu_{num_percentage}.csv')

if __name__ == '__main__':
    num_percentage = 50
    main(num_percentage)