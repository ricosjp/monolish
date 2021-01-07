""" Create HTML source """
import pandas

def df_to_html_table_caption(title:str, dataframe:pandas.DataFrame) -> str:
    """DataFrame to HTML caption and table"""
    if not dataframe.empty:
        table_html = df_to_html_table(dataframe)
        table_html = to_caption_on_html(title, table_html)
    else:
        table_html = ""
    return table_html

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

    html_tag = "<td>100</td>"
    html_tag_replace = (
        """
        <td>
            <div style='text-align:center; background: #c0c0c0'>
                <strong>100.0</strong>
            </div>
        </td>
        """
    )

    html_table = html_table.replace(html_tag, html_tag_replace)
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
</html>\n
    """
    return html
