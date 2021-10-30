import pandas as pd
import matchzoo


def _read_data(prd_path,
               rvw_path,
               rel_path,
               task,
               features_path):
    """
    Only read the review from year 2016
    """
    prd_table = pd.read_json(prd_path)
    rvw_table = pd.read_json(rvw_path)
    rel_table = pd.read_json(rel_path)

    prd = pd.DataFrame({
        'text_left': list(map(lambda x: f"{x[0]} {x[1]}", zip(prd_table['name'].tolist(), prd_table['description'].tolist()))),
        'id_left': prd_table['product_id'],
    })
    prd.id_left = prd.id_left.astype(str)
    prd = prd.reset_index(drop=True)

    rvw = pd.DataFrame({
        'text_right': rvw_table['content'],
        'id_right': rvw_table['review_id'],
        'time': rvw_table['time']
    })
    rvw.id_right = rvw.id_right.astype(str)
    rvw = rvw.reset_index(drop=True)

    rel_table['upvotes'] = rel_table['upvotes'].astype(int)
    rel: pd.DataFrame = pd.DataFrame({
        'id_left': rel_table['product_id'],
        'id_right': rel_table['review_id'],
        'label': rel_table['upvotes']
    })
    rel.id_left = rel.id_left.astype(str)
    rel = rel.reset_index(drop=True)
    return matchzoo.map_pack(prd, rvw, relation=rel, task=task)
