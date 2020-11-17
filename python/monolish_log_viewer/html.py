""" Create HTML source """
import pandas

def df_to_html_table(dataframe:pandas.DataFrame) -> str:
    """DataFrame to HTML table"""
    html_table = dataframe.to_html(justify="center")
    return html_table

def to_caption_on_html(title:str, html_table:str) -> str:
    """add caption"""
    caption = f"<caption>{title}</caption>"
    html_table = f"{caption}{html_table}"

    return html_table

def to_bold_on_html(html_table:str) -> str:
    """100.0 in bold and background"""

    html_table = html_table.replace(
        "<td>100</td>", "<td><div style='text-align:center; background: #c0c0c0'><strong>100.0</strong></div></td>")
    return html_table

def table_in_html(table_html:str) -> str:
    """create html source"""
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
