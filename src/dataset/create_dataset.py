import jsonlines
import numpy as np
from typing import List, Dict, Tuple, DefaultDict, Any
from collections import defaultdict
import time
import random
import pickle
import os
import concurrent.futures
from pathlib import Path
from tqdm.auto import tqdm
from dateutil.parser import parse, ParserError
import openai

try:
    import config
    openai.api_key = config.OPENAI_API_KEY
except ImportError:
    openai.api_key = os.environ.get('OPENAI_API_KEY')


from .settings import PATH_TO_RAW_DATA, PATH_TO_DATASET_PKL, PATH_TO_DATASET_DICT_PKL, EMBEDDING_MODEL, LEN_EMBEDDINGS

from .text_splitter import TokenSplitter, split_into_sentences



error_count_dict = {
    "Entry has no source.": 0,
    "Entry has no title.": 0,
    "Entry has no text.": 0,
    "Entry has no URL.": 0,
    "Entry has wrong citation level.": 0
}


class MissingDataException(Exception):
    pass


class Dataset:
    def __init__(self,
            jsonl_data_path: str = PATH_TO_RAW_DATA,  # Path to the dataset .jsonl file.
            custom_sources: List[str] = None,  # List of sources to include, like "alignment forum", "lesswrong", "arxiv",etc.
            rate_limit_per_minute: int = 3_500,  # Rate limit for the OpenAI API.
            min_tokens_per_block: int = 300, # Minimum number of tokens per block.
            max_tokens_per_block: int = 400, # Maximum number of tokens per block.
            embedding_batch_size: int = 200,  # Number of texts to embed at once.
            fraction_of_articles_to_use: float = 1.0,  # Fraction of articles to use. If 1.0, use all articles.
            path_to_dataset_pkl: str = PATH_TO_DATASET_PKL,  # Path to the dataset .pkl file.
            starting_article_index: int = -1  # Starting article index. If -1, start from the beginning.
        ):
        self.jsonl_data_path = jsonl_data_path
        self.custom_sources = custom_sources
        self.rate_limit_per_minute = rate_limit_per_minute
        self.delay_in_seconds = 60.0 / self.rate_limit_per_minute
        self.fraction_of_articles_to_use = fraction_of_articles_to_use
        self.path_to_dataset_pkl = path_to_dataset_pkl
        self.starting_article_index = starting_article_index
        
        self.min_tokens_per_block = min_tokens_per_block  # for the text splitter
        self.max_tokens_per_block = max_tokens_per_block  # for the text splitter
        self.embedding_batch_size = embedding_batch_size  # for the text splitter
        
        self.metadata: List[Tuple[str]] = []  # List of tuples, each containing the title, author, date, URL, and tags of an article.
        self.embedding_strings: List[str] = []  # List of strings, each being a few paragraphs from a single article (not exceeding max_tokens_per_block tokens).
        self.embeddings_metadata_index: List[int] = [] # List of integers, each being the index of the article from which the embedding string was taken.

        self.articles_count: DefaultDict[str, int] = defaultdict(int)  # Number of articles per source. E.g.: {'source1': 10, 'source2': 20, 'total': 30}

        if self.custom_sources is not None:
            for source in self.custom_sources:
                self.articles_count[source] = 0
        self.total_articles_count = 0
        
        self.total_char_count = 0
        self.total_word_count = 0
        self.total_sentence_count = 0
        self.total_block_count = 0
        
        self.sources_so_far: List[str] = []
        self.info_types: Dict[str, List[str]] = {}
        
        self.embeddings = None
    
    def extract_info_from_article(self, article: Dict[str, Any]) -> Tuple[str]:
        """
        This function extracts the title, author, date, URL, tags, and text from an article.
        
        Args:
            article (Dict[str, Any]): a dictionary containing the article's text and metadata.

        Returns:
            Tuple[str]: a tuple containing the title, author, date, URL, tags, and text of the article.
        """
        title: str = ""
        author: str = ""
        date_published: str = None
        url: str = None
        tags: str = None
        text: str = None
        
        # Get title
        if 'title' in article and 'book_title' in article and article['title']: title = article['title']
        elif 'book_title' in article and 'title' not in article and article['book_title']: 
            title = article['book_title']
        elif 'title' in article and article['title']: 
            title = article['title']
        title = title.strip('\n').replace('\n', ' ')[:100]

        # Get author
        if 'author' in article and 'authors' in article and article['author']: author = article['author']
        elif 'authors' in article and article['authors']: author = article['authors']
        elif 'author' in article and article['author']: author = article['author']
        if type(author) == str: author = get_authors_list(author)
        if type(author) == list: author = ', '.join(author)
        author = author.strip('\n').replace('\n', ' ')[:100]

        # Get date published
        if 'date_published' in article and article['date_published'] and len(article['date_published']) >= 10: date_published = article['date_published'][:10]
        elif 'published' in article and article['published'] and len(article['published']) >= 16: date_published = article['published'][:16]
        else: date_published = None
        if date_published is not None:
            date_published = standardize_date(date_published)
            
        # Get URL
        if 'link' in article and article['link']: url = article['link']
        elif 'url' in article and article['url']: url = article['url']
        elif 'doi' in article and article['doi']: url = article['doi']
        else: url = None
            
        # Get tags
        if 'tags' in article and article['tags']:
            if type(article['tags']) == list: tags = ', '.join([val['term'] for val in article['tags']])
            elif type(article['tags']) == str: tags = article['tags']
            else: tags = None
        
        # Get text
        if 'text' in article and article['text']: text = article['text']
        else:
            raise MissingDataException(f"Entry has no text.")

        return (title, author, date_published, url, tags, text)
           
    def get_alignment_texts(self):
        start = time.time()
        
        text_splitter = TokenSplitter(self.min_tokens_per_block, self.max_tokens_per_block)
        with jsonlines.open(self.jsonl_data_path, "r") as reader:
            for entry_idx, entry in enumerate(tqdm(reader)):
                try:
                    if 'source' not in entry: 
                        if 'url' in entry and entry['url'] == "https://www.cold-takes.com/": 
                            entry["source"] = "Cold Takes"
                        elif 'question' in entry and 'answer' in entry: 
                            entry["source"] = "printouts"
                            continue # for now, skip printouts
                        elif 'article_url' in entry and entry['article_url'] == "https://www.gwern.net":
                            entry["source"] = "gwern.net"
                        elif 'url' in entry and entry['url'] == "https://generative.ink/posts/":
                            entry["source"] = "generative.ink"
                        elif 'url' in entry and entry['url'][:24] == "https://greaterwrong.com":
                            entry["source"] = "greaterwrong.com"
                        else:
                            raise MissingDataException("Entry has no source.")
                    
                    # if we specified custom sources, only include articles from those sources
                    if (self.custom_sources is not None) and (entry['source'] not in self.custom_sources):
                        continue

                    """if entry["source"] == 'alignment forum':
                        if int(entry['score'].replace('−', '-')) < 70: continue
                    elif entry["source"] == 'lesswrong':
                        if int(entry['score'].replace('−', '-')) < 150: continue
                    elif entry["source"] == 'arxiv':
                        if 'citation_level' != '0': continue

                    desired_source_proportions = {
                        "https://aipulse.org": 1,
                        "ebook": 0,
                        "https://qualiacomputing.com": 0.02,
                        "alignment forum": .7,
                        "lesswrong": .5,
                        "manual": 1,
                        "arxiv": 0.1,
                        "https://deepmindsafetyresearch.medium.com/": 1,
                        "waitbutwhy.com": 1,
                        "GitHub": 1,
                        "https://aiimpacts.org": 0.2,
                        "arbital.com": 0.2,
                        "carado.moe": 0.3,
                        "nonarxiv_papers": 0.1,
                        "https://vkrakovna.wordpress.com": .5,
                        "https://jsteinhardt.wordpress.com": .5,
                        "audio-transcripts": 0.2,
                        "https://intelligence.org": .1,
                        "youtube": 0.07,
                        "reports": 0.4,
                        "https://aisafety.camp": 1,
                        "curriculum": 1,
                        "https://www.yudkowsky.net": 0.2,
                        "distill": 1,
                        "Cold Takes": 0.5,
                        "printouts": 1,
                        "gwern.net": 1,
                        "generative.ink": 1,
                        "greaterwrong.com": 0.2
                    }
                    
                    random_number = random.random()
                    if random_number > desired_source_proportions[entry['source']]:
                        continue
                    """
                    # if we specified a fraction of articles to use, only use that fraction from the remaining articles
                    random_number = random.random()
                    if random_number > self.fraction_of_articles_to_use:
                        continue

                    
                    title, author, date_published, url, tags, text = self.extract_info_from_article(entry)
                    
                    if (((title or '').strip() == '') + ((author or '').strip() == '') + ((url or '').strip() == '')) > 1:continue  # If there's less than 2 of 'title', 'author' and 'url', ignore this text
                    if len(text) < 500:continue  # If the text is too short, ignore this text
                    
                    signature = ""
                    if title: signature += f"Title: {title}; "
                    else: signature += f"Title: None; "
                    if author: signature += f"Author: {author}"
                    else: signature += f"Author: None"
                    signature = signature.replace("\n", " ")
                    
                    # We're keeping the text so we inc the aticle count
                    self.articles_count[entry['source']] += 1
                    self.total_articles_count += 1
                    
                    if self.total_articles_count % 1000 == 0:
                        print(f"\n{self.total_articles_count} articles in {time.time() - start:.2f} seconds.")
                                        
                    # Add info to metadata and embedding strings, and update counts
                    self.metadata.append((title, author, date_published, url, tags))
                    blocks = text_splitter.split(text, signature)
                    self.embedding_strings.extend(blocks)
                    self.embeddings_metadata_index.extend([self.total_articles_count-1] * len(blocks))
                    
                    self.total_char_count += len(text)
                    self.total_word_count += len(text.split())
                    self.total_sentence_count += len(split_into_sentences(text))
                    self.total_block_count += len(blocks)
                    
                except MissingDataException as e:
                    if str(e) not in error_count_dict:
                        error_count_dict[str(e)] = 0
                    error_count_dict[str(e)] += 1
        
        print(f"\nArticle count: {len(self.metadata)}")
        print(f"Total char count: {self.total_char_count}")
        print(f"Total word count: {self.total_word_count}")
        print(f"Total sentence count: {self.total_sentence_count}")
        print(f"Total block count: {self.total_block_count}")
        print(f"Total time: {time.time() - start:.2f} seconds")
        
        self.embeddings = np.zeros((len(self.embedding_strings), LEN_EMBEDDINGS), dtype=np.float32)
        self.get_embeddings()
        self.save_data(self.path_to_dataset_pkl)
        
    def get_embeddings(self, start_idx: int = 0):
        def get_embeddings_at_index(texts: str, batch_idx: int, batch_size: int = 200): # int, np.ndarray
            embeddings = np.zeros((batch_size, 1536))
            openai_output = openai.Embedding.create(
                model=EMBEDDING_MODEL, 
                input=texts
            )['data']
            for i, embedding in enumerate(openai_output):
                embeddings[i] = embedding['embedding']
            return batch_idx, embeddings

        batch_size = self.embedding_batch_size
        rate_limit = self.rate_limit_per_minute / 60  # Maximum embeddings per second
        save_interval = 10000
        
        start = time.time()
        
        # if self.embeddings is None:
        #     self.embeddings = np.zeros((len(self.embedding_strings), LEN_EMBEDDINGS), dtype=np.float32)
        # else:
        #     self.embeddings = np.resize(self.embeddings, (len(self.embedding_strings), LEN_EMBEDDINGS))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                get_embeddings_at_index, 
                self.embedding_strings[batch_idx:batch_idx+batch_size], 
                batch_idx,
                len(self.embedding_strings[batch_idx:batch_idx+batch_size])
            ) for batch_idx in range(start_idx, len(self.embedding_strings), batch_size)]
            num_completed = 0
            for future in concurrent.futures.as_completed(futures):
                batch_idx, embeddings = future.result()
                num_completed += embeddings.shape[0]
                self.embeddings[batch_idx:batch_idx+embeddings.shape[0]] = embeddings

                if num_completed % save_interval == 0:
                    if not np.any(self.embeddings == 0):
                        self.save_embeddings('embeddings', num_completed - save_interval, num_completed)
                    else:
                        print(f"Skipped saving embeddings because there are still zeros in the embeddings array.")
                        time.sleep(10)
                        self.save_embeddings('embeddings', num_completed - save_interval, num_completed)
                
                elapsed_time = time.time() - start
                expected_time = num_completed / rate_limit
                sleep_time = max(expected_time - elapsed_time, 0)
                time.sleep(sleep_time)

                print(f"Completed {num_completed}/{len(self.embedding_strings)-start_idx} embeddings in {elapsed_time:.2f} seconds.")
        
        if num_completed % save_interval != 0:
            last_saved_idx = num_completed - (num_completed % save_interval)
            self.save_embeddings('embeddings', last_saved_idx, num_completed)

        print()

    def save_embeddings(self, file_prefix, start_idx, end_idx):
        file_name = f"{file_prefix}_{start_idx}-{end_idx}.npy"
        np.save(file_name, self.embeddings[start_idx:end_idx])
        print(f"Saved embeddings to file: {file_name}")

    
    def save_data(self, path: str = PATH_TO_DATASET_DICT_PKL):
        # Save the data to a pickle file
        print(f"Saving data to {path}...")
        data = {
            "metadata": self.metadata,
            "embedding_strings": self.embedding_strings,
            "embeddings_metadata_index": self.embeddings_metadata_index,
            "embeddings": self.embeddings.astype(np.float32) if type(self.embeddings) is np.ndarray else None,
            "articles_count": self.articles_count,
            "total_articles_count": self.total_articles_count,
            "total_char_count": self.total_char_count,
            "total_word_count": self.total_word_count,
            "total_sentence_count": self.total_sentence_count,
            "total_block_count": self.total_block_count,
            "starting_article_index": self.starting_article_index
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_data(self, path: str, starting_article_index: int = 0):
        """
        Load the data from a pickle file.
        
        Args:
            path (str): The path to the pickle file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.metadata = data["metadata"]
        self.embedding_strings = data["embedding_strings"]
        self.embeddings_metadata_index = data["embeddings_metadata_index"]
        self.embeddings = data["embeddings"]
        self.articles_count = data["articles_count"]
        self.total_articles_count = data["total_articles_count"]
        self.total_char_count = data["total_char_count"]
        self.total_word_count = data["total_word_count"]
        self.total_sentence_count = data["total_sentence_count"]
        self.total_block_count = data["total_block_count"]
        if 'starting_article_index' in data:
            self.starting_article_index = data["starting_article_index"]
        else:
            self.starting_article_index = starting_article_index

def get_authors_list(authors_string: str) -> List[str]:
    """
    Given a string of authors, return a list of the authors, even if the string contains a single author.
    """
    authors_string = authors_string.replace(" and ", ",")
    authors_string = authors_string.replace('\n', ' ')
    authors = []
    if authors_string is None:
        return []
    if "," in authors_string:
        authors = [author.strip() for author in authors_string.split(",")]
    else:
        authors = [authors_string.strip()]
    return authors

def standardize_date(date_string, default_date='n/a'):
    try:
        dt = parse(date_string)
        return dt.strftime('%Y-%m-%d')
    except (ParserError, ValueError):
        return default_date



"""
if __name__ == "__main__":
    # List of possible sources:
    all_sources = ["https://aipulse.org", "ebook", "https://qualiacomputing.com", "alignment forum", "lesswrong", "manual", "arxiv", "https://deepmindsafetyresearch.medium.com", "waitbutwhy.com", "GitHub", "https://aiimpacts.org", "arbital.com", "carado.moe", "nonarxiv_papers", "https://vkrakovna.wordpress.com", "https://jsteinhardt.wordpress.com", "audio-transcripts", "https://intelligence.org", "youtube", "reports", "https://aisafety.camp", "curriculum", "https://www.yudkowsky.net", "distill", "Cold Takes", "printouts", "gwern.net", "generative.ink", "greaterwrong.com"] # These sources do not have a source field in the .jsonl file

    # List of sources we are using for the test run:
    custom_sources = [
        # "https://aipulse.org", 
        # "ebook", 
        # "https://qualiacomputing.com", 
        # "alignment forum", 
        # "lesswrong", 
        "manual", 
        # "arxiv", 
        # "https://deepmindsafetyresearch.medium.com", 
        "waitbutwhy.com", 
        # "GitHub", 
        # "https://aiimpacts.org", 
        # "arbital.com", 
        # "carado.moe", 
        # "nonarxiv_papers", 
        # "https://vkrakovna.wordpress.com", 
        "https://jsteinhardt.wordpress.com", 
        # "audio-transcripts", 
        # "https://intelligence.org", 
        # "youtube", 
        # "reports", 
        "https://aisafety.camp", 
        "curriculum", 
        "https://www.yudkowsky.net", 
        # "distill",
        # "Cold Takes",
        # "printouts",
        # "gwern.net",
        # "generative.ink",
        # "greaterwrong.com"
    ]
    
    dataset = Dataset(
        jsonl_data_path=PATH_TO_RAW_DATA.resolve(), 
        custom_sources=custom_sources, 
        rate_limit_per_minute=3500, 
        min_tokens_per_block=200, max_tokens_per_block=300, 
        # fraction_of_articles_to_use=1/2000
    )
    dataset.get_alignment_texts()
    dataset.get_embeddings()
    # dataset.save_embeddings("data/embeddings.npy")
    
    dataset.save_class(PATH_TO_DATASET.resolve())
    # # dataset = pickle.load(open("dataset.pkl", "rb"))
    """
