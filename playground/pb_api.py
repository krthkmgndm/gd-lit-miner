from pymed import PubMed 

pubmed = PubMed()

query = input('What is your query? ')
max_results = 5 

results = pubmed.query(query, max_results=max_results)

for article in results:
    title = article.title
    authors = [author['lastname'] for author in article.authors] if article.authors else "Unknown"
    pub_date = article.publication_date
    abstract = article.abstract

    print("-" * 80)
    print(f"Title: {title}")
    print(f"Authors: {', '.join(authors)}")
    print(f"Publication Date: {pub_date}")
    print(f"Abstract: {abstract}")
