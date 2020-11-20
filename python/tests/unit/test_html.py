""" test html """
import numpy
import pandas
from monolish_log_viewer import html

def test_df_to_html_table():
    """ test df_to_html_table """
    # base data
    dataframe = pandas.DataFrame(
        numpy.arange(12).reshape(3, 4),
        columns = ["col_0", "col_1", "col_2", "col_3"],
        index = ["row_0", "row_1", "row_2"]
    )

    # html.df_to_html_table
    html_table = html.df_to_html_table(dataframe)
    html_table = html_table.replace("\n", "").replace(" ", "")

    # create HTML
    cols = dataframe.columns
    idxs = dataframe.index

    header_html = '<thead>'
    header_html = header_html + '<tr style="text-align: center;">'
    header_html = header_html + '<th></th>'
    for col in cols:
        header_html = header_html + f'<th>{col}</th>'
    header_html = header_html + '</tr>'
    header_html = header_html + '</thead>'

    record_html = '<tbody>'
    for idx in idxs:
        record_html = record_html + '<tr>'
        record_html = record_html + f'<th>{idx}</th>'
        for col in cols:
            record_html = record_html + f'<td>{dataframe[col][idx]}</td>'
        record_html = record_html + '</tr>'
    record_html = record_html + '</tbody>'

    html_text = f"""
        <table border="1" class="dataframe">
        {header_html}
        {record_html}
        </table>\n
    """
    html_text = html_text.replace("\n", "").replace(" ", "")

    assert html_table == html_text
