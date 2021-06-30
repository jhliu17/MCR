import pandas as pd
import matchzoo


def upvotes_map_func(upvote):
    if upvote >= 16:
        return 4

    if upvote >= 8:
        return 3

    if upvote >= 4:
        return 2

    if upvote >= 2:
        return 1

    if upvote == 1:
        return 0

    return -1


def _read_data(prd_path,
               rvw_path,
               rel_path,
               task,
               features_path):
    """Read the json file with multiple intervel training.
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
        'id_right': rvw_table['review_id']
    })
    rvw.id_right = rvw.id_right.astype(str)
    rvw = rvw.reset_index(drop=True)

    rel_table['upvotes'] = rel_table['upvotes'].astype(int)

    # mapping upvotes number into different categories
    rel_table.upvotes = rel_table.upvotes.apply(upvotes_map_func)

    rel: pd.DataFrame = pd.DataFrame({
        'id_left': rel_table['product_id'],
        'id_right': rel_table['review_id'],
        'label': rel_table['upvotes']
    })
    rel.id_left = rel.id_left.astype(str)
    rel = rel.reset_index(drop=True)

    # remove data with 0 upvote (label == -1) as noise
    zero_upvote_index = rel[rel.label == -1].index
    rel.drop(index=zero_upvote_index, inplace=True)

    return matchzoo.map_pack(prd, rvw, relation=rel, task=task)
