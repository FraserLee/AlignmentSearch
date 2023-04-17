
import React, { useState } from "react";

const Footer: React.FC = () => {
  const [isContentVisible, setIsContentVisible] = useState(false);

  const toggleContentVisibility = () => {
    setIsContentVisible(!isContentVisible);
  };

  return (
    <footer>

      <button
        className="text-gray-500 hover:text-gray-700 hover:underline"
        onClick={toggleContentVisibility}      >
        {isContentVisible ? "hide about" : "about this conversational agent"} 
      </button>

      {isContentVisible && (
        <div className="text-gray-500">
          <p>
            This site has been created by McGill students Henri Lemoine, Fraser
            Lee, and Thomas Lemoine as an attempt to create a "conversational
            FAQ" that can answer questions about AI alignment. When asked a
            question, we
          </p>
          <ol>
            <li>Embed the question into a low dimensional semantic space</li>
            <li>
              Pull the semantically closest passages out of a massive alignment
              dataset
            </li>
            <li>
              Instruct an LLM to construct a response while citing these
              passages
            </li>
            <li>
              Display this response in conversational flow with inline
              citations
            </li>
          </ol>
          <p>
            We've created this as an attempt on the{" "}
            <a href="https://www.lesswrong.com/posts/SLRLuiuDykfTdmesK/speed-running-everyone-through-the-bad-alignement-bingo">
              $5k bounty for a LW conversational agent
            </a>
            .
          </p>
          <p>
            We hope that it can be a useful tool for the community, and we're
            eager to hear feedback and suggestions. For a technical report on
            our implementation, see our{" "}
            <a href="https://www.lesswrong.com/posts/bGn9ZjeuJCg7HkKBj/introducing-alignmentsearch-an-ai-alignment-informed">
              LessWrong post
            </a>
            .
          </p>
        </div>
      )}
    </footer>
  );
};

export default Footer;