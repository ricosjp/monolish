#!/usr/bin/env python3
# coding: utf-8
"""HTML utilities suitable."""

"""list of HTML table to HTML"""
def create_table(title, columns_list, aggr_ndarray) -> str:
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

"""list of HTML table to HTML"""
def create_html(html_tables) -> str:
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
