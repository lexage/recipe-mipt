
GET_DOCUMENTS_QUERY = '''
SELECT d.id, d.name, d.content, 
    s.name as section, l.name as library
FROM documents d
JOIN sections s ON d.section_id = s.id
JOIN libraries l ON s.library_id = l.id
'''

GET_EXAMPLES_QUERY = '''
SELECT e.id, e.order_id, e.doc_id, e.content
FROM examples e
'''