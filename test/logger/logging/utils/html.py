#!/usr/bin/env python3
# coding: utf-8
"""HTML utilities suitable for global use."""

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

def create_html(html_tables) -> str:
    # join html tables
    html_tables = ','.join(html_tables)
    html_tables = html_tables.replace(",", "")

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