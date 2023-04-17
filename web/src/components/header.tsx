import React from "react";
import Link from "next/link";
import About from "./about";

const Header: React.FC<{ page: "index" | "semantic" }> = ({ page }) => {
    const sidebar = page === "index" ? (
        <span className="flex flex-col flex-1 justify-start text-right mt-4">
            <p className="my-0">Conversational Agent</p>
            <Link href="/semantic">Semantic Search</Link>
        </span>
    ) : (
        <span className="flex flex-col flex-1 justify-start text-right">
            <Link href="/">Conversational Agent</Link>
            <p className="my-0">Semantic Search</p>
        </span>

    );

    return (<div className="my-20">

        <div className="flex">
            <h1 className="flex-1 text-4xl"> Alignment Search </h1>
            {sidebar}
        </div>
        <About />

    </div>);
};

export default Header;
