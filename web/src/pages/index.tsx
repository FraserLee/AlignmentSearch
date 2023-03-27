import { type NextPage } from "next";
import React from "react";
import { useState } from "react";
import Head from "next/head";

const Home: NextPage = () => {
    return (
        <>
            <Head>
                <title>Alignment Search</title>
            </Head>
            <main>
                <h1>Alignment Search</h1>
                <p>
                    This site is an attempt on the <a href="https://www.lesswrong.com/posts/SLRLuiuDykfTdmesK/speed-running-everyone-through-the-bad-alignement-bingo">
                        $5k bounty for a LW conversational agent

                    </a>, created by Henri Lemoine, Fraser Lee and Thomas Lemoine.
                </p>
                <p>
                    We will soon embed the entirety of the alignment dataset,
                    separating it into chunks of ~500 tokens for comparing 
                    semantic similarity between query and paragraph/chunk. This 
                    may cost anywhere from 20$ to 500$ based on our estimates 
                    (we will soon sample the dataset to improve our estimate), 
                    so if anyone else is considering this, you can message us to 
                    coordinate sharing the embeddings to avoid redundancy.
                </p>

                <p className="mt-4">Get the most semantic similar results to a query:</p>
                <SearchBox />
            </main>
        </>
    );
};

// Round trip test. If this works, our heavier usecase probably will (famous last words)
// The one real difference is we'll want to send back a series of results as we get
// them back from OpenAI - I think we can just do this with a websocket, which
// shouldn't be too much harder.

const SearchBox: React.FC = () => {

    const [query,   setQuery]   = useState("");
    const [results, setResults] = useState<{title: string, author: string, date: string, url: string, tags: string, text: string}[]>([]);
    const [loading, setLoading] = useState(false);

    const semantic_search = async (query: String) => {
        
        setLoading(true);

        const res = await fetch("/api/semantic_search", {
            method: "POST",
            headers: { "Content-Type": "application/json", },
            body: JSON.stringify({query: query}),
        })

        const data = await res.json();

        setLoading(false);

        // data looks like 
        // { 0: "{'title': 'First Title', 'author': 'Bob Miles', 'date': 'March 1st, 2023', 'url': 'https://example.com', 'tags': ['tag1', 'tag2'], 'text': 'This is the content of the article'}",
        //   1: "{'title': 'Second Title', 'author': 'Frank Ocean', 'date': 'March 6th, 2023', 'url': 'https://ai.com', 'tags': ['tag3', 'tag4'], 'text': 'This is the content of the article'}"
        // }
        // so we need to convert it to a list of objects

        return Object.keys(data).map((key) => JSON.parse(data[key])) || [{title: "error", author: "error", date: "error", url: "error", tags: ["error"], text: "error"}];
    };

    return (
        <>
            <form className="flex mb-2" onSubmit={async (e) => { // store in a form so that <enter> submits
                e.preventDefault();
                setResults(await semantic_search(query));
            }}>

                <input
                    type="text"
                    className="border border-gray-300 px-1 flex-1"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <button className="ml-2" type="submit" disabled={loading}>
                    {loading ? "Loading..." : "Search"}
                </button>
            </form>

            {loading ? <p>loading...</p> : ( // display results in list
                <ul>
                    {results.map((result) => (
                        <li key={result.url} className="my-1">
                            <a href={result.url}>{result.text}</a>
                        </li>
                    ))}
                </ul>
            )}
        </>
    );
};


export default Home;
