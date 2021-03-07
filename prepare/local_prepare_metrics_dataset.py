import pandas
import mmh3

D = '0123456789abcdef'


def _mmh3_guava_to_string(v):
    """
    Guava's hash toString()
    """
    b_ = mmh3.hash_bytes(v)
    return "".join([f"{D[(b >> 4) & 0xf]}{D[b & 0xf]}" for b in b_])


def _keys():
    tags = {'cat', 'forest', 'bee','bear', 'doctor','sunset', 'apple', 'woodland'}
    kw = pandas.read_csv('/Users/bohdans/unsplash/lite/keywords.tsv000', sep='\t')
    df = kw[(kw['keyword'].isin(tags)) & (kw['ai_service_1_confidence'] > 50.0)]
    df['uid'] = df['photo_id'].transform(_mmh3_guava_to_string)
    del df['ai_service_1_confidence']
    del df['ai_service_2_confidence']
    del df['suggested_by_user']
    del df['photo_id']
    return df


df_ = pandas.read_csv('/Users/bohdans/unsplash/lite/collections.tsv000', sep='\t')
df_['uid'] = df_['photo_id'].transform(_mmh3_guava_to_string)
del df_['collection_title']
del df_['photo_collected_at']

df_.to_csv('/Users/bohdans/.stash/dataset_002/collections.csv')

_keys().to_csv('/Users/bohdans/.stash/dataset_002/keys.csv')