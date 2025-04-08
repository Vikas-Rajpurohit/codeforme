import spacy
import re
from atlassian import Confluence
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")

def html_table_to_text(html_table):
    soup = BeautifulSoup(html_table, "html.parser")

    # Extract table rows
    rows = soup.find_all("tr")

    # Determine if the table has headers or not
    has_headers = any(th for th in soup.find_all("th"))

    # Extract table headers, either from the first row or from the <th> elements
    if has_headers:
        headers = [th.get_text(strip=True) for th in soup.find_all("th")]
        row_start_index = 1  # Skip the first row, as it contains headers
    else:
        first_row = rows[0]
        headers = [cell.get_text(strip=True) for cell in first_row.find_all("td")]
        row_start_index = 1

    # Iterate through rows and cells, and use NLP to generate sentences
    text_rows = []
    for row in rows[row_start_index:]:
        cells = row.find_all("td")
        cell_sentences = []
        for header, cell in zip(headers, cells):
            # Generate a sentence using the header and cell value
            doc = nlp(f"{header}: {cell.get_text(strip=True)}")
            sentence = " ".join([token.text for token in doc if not token.is_stop])
            cell_sentences.append(sentence)

        # Combine cell sentences into a single row text
        row_text = ", ".join(cell_sentences)
        text_rows.append(row_text)

    # Combine row texts into a single text
    text = "nn".join(text_rows)
    return text

def html_list_to_text(html_list):
    soup = BeautifulSoup(html_list, "html.parser")
    items = soup.find_all("li")
    text_items = []
    for item in items:
        item_text = item.get_text(strip=True)
        text_items.append(f"- {item_text}")
    text = "n".join(text_items)
    return text

def process_html_document(html_document):

    soup = BeautifulSoup(html_document, "html.parser")

    # Replace tables with text using html_table_to_text
    for table in soup.find_all("table"):
        table_text = html_table_to_text(str(table))
        table.replace_with(BeautifulSoup(table_text, "html.parser"))

    # Replace lists with text using html_list_to_text
    for ul in soup.find_all("ul"):
        ul_text = html_list_to_text(str(ul))
        ul.replace_with(BeautifulSoup(ul_text, "html.parser"))

    for ol in soup.find_all("ol"):
        ol_text = html_list_to_text(str(ol))
        ol.replace_with(BeautifulSoup(ol_text, "html.parser"))

    # Replace all types of <br> with newlines
    br_tags = re.compile('<br>|<br/>|<br />')
    html_with_newlines = br_tags.sub('n', str(soup))

    # Strip remaining HTML tags to isolate the text
    soup_with_newlines = BeautifulSoup(html_with_newlines, "html.parser")

    return soup_with_newlines.get_text()

# Extract the content in the "storage" format
storage_value = page_content['body']['storage']['value']

# Clean the HTML tags to get the text content
text_content = process_html_document(storage_value)