import pandas as pd

def create_table(title, columns_list, aggr_ndarray) -> str:
    """list of HTML table to HTML"""
    # caption
    caption = f"<caption>{title}</caption>"
    # table header
    column_html = "".join(list(map(lambda column:f"<th>{column}</th>", columns_list)))
    table_header = f"<tr>{column_html}</tr>"
    # table data
    table_data = "".join(list(map(lambda aggr_narray:"<tr>" + "".join(list(map(lambda elem:f'<td>{str(elem)}</td>', aggr_narray))) + "</tr>", aggr_ndarray)))
    # table
    table = caption + table_header + table_data
    table = f"<table border='1'>{table}</table>"
    return table

def df_to_html_table(df):
    html_table = df.to_html()
    return html_table

def to_bold_on_html(html_table):
    """100.0 in bold and background"""
    html_table = html_table.replace("100.0", "<div style='text-align:center; background: #c0c0c0'><strong>100.0</strong></div>")
    return html_table

def table_in_html(html_table) -> str:
    html = f"""
        <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="utf-8">
            </head>
            <body>
                {table_html}
            </body>
        </html>
    """
    return html

def create_html(html_tables) -> str:
    """list of HTML table to HTML"""
    # join html tables
    html_tables = ','.join(html_tables)
    html_tables = html_tables.replace(",", "")
    # 100.0 in bold
    html_tables = html_tables.replace("100.0", "<div style='text-align:center; background: #c0c0c0'><strong>100.0</strong></div>")

    # create html
    html = f"""
        <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="utf-8">
            </head>
            <body>
                {html_tables}
            </body>
        </html>
    """
    return html
