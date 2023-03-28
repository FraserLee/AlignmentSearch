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
                {/* <StreamButton /> */}
            </main>
        </>
    );
};

// Round trip test. If this works, our heavier usecase probably will (famous last words)
// The one real difference is we'll want to send back a series of results as we get
// them back from OpenAI - I think we can just do this with a websocket, which
// shouldn't be too much harder.

type SemanticEntry = {
    title: string;
    author: string;
    date: string;
    url: string;
    tags: string;
    text: string;
};

const ShowSemanticEntry: React.FC<{entry: SemanticEntry}> = ({entry}) => {
    return (
        <div className="my-3">

            {/* horizontally split first row, title on left, author on right */}
            <div className="flex">
                <h3 className="text-xl flex-1">{entry.title}</h3>
                <p className="flex-1 text-right my-0">{entry.author} - {entry.date}</p>
            </div>

            <p className="text-sm">{entry.text}</p>

            <a href={entry.url}>Read more</a>
        </div>
    );
};

const SearchBox: React.FC = () => {

    const [query,   setQuery]   = useState("");
    const [results, setResults] = useState<SemanticEntry[] | string>([]);
    const [loading, setLoading] = useState(false);

    const semantic_search = async (query: String) => {
        
        setLoading(true);

        const res = await fetch("/api/semantic_search", {
            method: "POST",
            headers: { "Content-Type": "application/json", },
            body: JSON.stringify({query: query}),
        })

        if (!res.ok) {
            setLoading(false);
            return "load failure: " + res.status;
        }

        const data = await res.json();
        setLoading(false);
        return data;
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

            {
                loading ? <p>loading...</p> :
                typeof results === "string" ? <p className="text-red-500">{results}</p> :
                <ul>
                    {results.map((result, i) => (
                        <li key={i}>
                            <ShowSemanticEntry entry={result} />
                        </li>
                    ))}
                </ul>
            }
        </>
    );
};


// a button which, when pressed, opens a SSE to /api/stream and gradually
// fills in results in a <p> below

// const StreamButton: React.FC = () => {
//
//     const [streaming, setStreaming] = useState(false);
//     const [results, setResults] = useState<string[]>([]);
//
//     const start_stream = async () => {
//         setStreaming(true);
//
//         const res = await fetch("/api/stream", {
//             method: "GET",
//             headers: { "Content-Type": "application/json", },
//         })
//
//         if (!res.ok) {
//             setStreaming(false);
//             return "load failure: " + res.status;
//         }
//
//         const reader = res.body!.getReader();
//         let decoder = new TextDecoder();
//
//         while (true) {
//             const { done, value } = await reader.read();
//             if (done) {
//                 break;
//             }
//             const text = decoder.decode(value);
//             setResults((prev) => [...prev, text]);
//         }
//
//         setStreaming(false);
//
//     };
//
//     return (
//         <>
//             <button className="ml-2" type="submit" disabled={streaming} onClick={start_stream}>
//                 {streaming ? "Streaming..." : "Stream"}
//             </button>
//
//             <p>
//                 {results.map((result, i) => (
//                     <span key={i}>{result}</span>
//                 ))}
//             </p>
//         </>
//     );
// };






export default Home;
