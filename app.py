import React, { useState } from 'react';
import { Send, Upload, Monitor, Bot, ExternalLink, RefreshCw } from 'lucide-react';

// Main application component
const App = () => {
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [uploadStatus, setUploadStatus] = useState('');
    const [scrapeUrl, setScrapeUrl] = useState('');
    const [scrapeInstructions, setScrapeInstructions] = useState('');
    const [scrapeResult, setScrapeResult] = useState('');
    const [isThinking, setIsThinking] = useState(false);

    // Placeholder for API key, which will be provided by the runtime
    const API_KEY = "";
    const API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent";

    // Handles the chat message submission
    const handleSendMessage = async () => {
        if (inputValue.trim() === '') return;

        const newUserMessage = { sender: 'user', text: inputValue };
        setMessages(prevMessages => [...prevMessages, newUserMessage]);
        setInputValue('');
        setIsThinking(true);

        try {
            // Placeholder for an actual LLM call.
            // For this example, we'll just simulate a response.
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            const nexiReply = `Nexi received: "${inputValue}"`;
            const newNexiMessage = { sender: 'nexi', text: nexiReply };
            setMessages(prevMessages => [...prevMessages, newNexiMessage]);

        } catch (error) {
            const errorMessage = { sender: 'system', text: `Error: ${error.message}` };
            setMessages(prevMessages => [...prevMessages, errorMessage]);
        } finally {
            setIsThinking(false);
        }
    };

    // Placeholder for file upload
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        setUploadStatus('Uploading...');
        // Simulate an upload process
        await new Promise(resolve => setTimeout(resolve, 2000));
        setUploadStatus(`Success: File "${file.name}" uploaded.`);
    };

    // Placeholder for web scraping
    const handleScrape = async () => {
        if (scrapeUrl.trim() === '') return;
        setScrapeResult('Scraping...');

        // Simulate a scraping process
        await new Promise(resolve => setTimeout(resolve, 3000));

        const result = `Scraped data from "${scrapeUrl}" with instruction: "${scrapeInstructions}"`;
        setScrapeResult(`Result: ${result}`);
    };

    return (
        <div className="bg-gray-900 text-white min-h-screen font-inter flex flex-col items-center p-4 sm:p-8">
            <div className="w-full max-w-5xl flex flex-col md:flex-row gap-8">
                {/* Main Console & Chat */}
                <div className="flex-1 bg-gray-800 rounded-3xl shadow-xl border border-gray-700 overflow-hidden">
                    <div className="bg-gray-700 p-4 border-b border-gray-600 flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <Monitor size={24} className="text-blue-400" />
                            <h1 className="text-xl font-bold text-blue-400">Nexi Console</h1>
                        </div>
                    </div>

                    {/* Chat Messages */}
                    <div className="p-6 h-[500px] overflow-y-auto space-y-4 text-sm custom-scrollbar">
                        {messages.map((msg, index) => (
                            <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[70%] p-3 rounded-2xl ${
                                    msg.sender === 'user'
                                    ? 'bg-blue-600 text-white'
                                    : msg.sender === 'nexi'
                                    ? 'bg-gray-700 text-gray-200 border border-gray-600'
                                    : 'bg-red-800 text-red-100 border border-red-600'
                                }`}>
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                        {isThinking && (
                            <div className="flex justify-start">
                                <div className="max-w-[70%] p-3 rounded-2xl bg-gray-700 text-gray-400 border border-gray-600">
                                    <RefreshCw size={16} className="inline-block animate-spin mr-2" />
                                    Nexi is thinking...
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Chat Input */}
                    <div className="p-4 border-t border-gray-600">
                        <div className="flex space-x-2">
                            <input
                                type="text"
                                className="flex-1 p-3 bg-gray-700 text-gray-200 rounded-xl border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyPress={(e) => { if (e.key === 'Enter') handleSendMessage(); }}
                                placeholder="Message Nexi..."
                                disabled={isThinking}
                            />
                            <button
                                onClick={handleSendMessage}
                                className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-50"
                                disabled={isThinking || inputValue.trim() === ''}
                            >
                                <Send size={20} />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right-hand side panels */}
                <div className="w-full md:w-80 flex flex-col gap-8">
                    {/* File Upload Panel */}
                    <div className="bg-gray-800 p-6 rounded-3xl shadow-xl border border-gray-700">
                        <h2 className="text-lg font-bold mb-4 flex items-center space-x-2">
                            <Upload size={20} className="text-green-400" />
                            <span>File Upload</span>
                        </h2>
                        <label className="block w-full text-sm text-gray-400 cursor-pointer p-4 border-2 border-dashed border-gray-600 rounded-xl hover:border-blue-500 transition-colors">
                            <input type="file" onChange={handleFileUpload} className="hidden" />
                            <span>Click to upload a file</span>
                        </label>
                        {uploadStatus && (
                            <div className="mt-4 p-3 text-xs rounded-xl bg-gray-700 text-green-400 border border-gray-600">
                                {uploadStatus}
                            </div>
                        )}
                    </div>

                    {/* Web Scraping Panel */}
                    <div className="bg-gray-800 p-6 rounded-3xl shadow-xl border border-gray-700">
                        <h2 className="text-lg font-bold mb-4 flex items-center space-x-2">
                            <ExternalLink size={20} className="text-purple-400" />
                            <span>Web Scrape</span>
                        </h2>
                        <div className="space-y-4">
                            <input
                                type="text"
                                className="w-full p-3 bg-gray-700 text-gray-200 rounded-xl border border-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                                placeholder="Target URL"
                                value={scrapeUrl}
                                onChange={(e) => setScrapeUrl(e.target.value)}
                            />
                            <textarea
                                className="w-full p-3 h-24 bg-gray-700 text-gray-200 rounded-xl border border-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                                placeholder="Instructions (e.g., 'Summarize the article')"
                                value={scrapeInstructions}
                                onChange={(e) => setScrapeInstructions(e.target.value)}
                            />
                            <button
                                onClick={handleScrape}
                                className="w-full p-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition-colors disabled:opacity-50"
                                disabled={scrapeUrl.trim() === ''}
                            >
                                Scrape
                            </button>
                        </div>
                        {scrapeResult && (
                            <div className="mt-4 p-3 text-xs rounded-xl bg-gray-700 text-purple-400 border border-gray-600 overflow-auto max-h-24">
                                {scrapeResult}
                            </div>
                        )}
                    </div>
                </div>
            </div>
            {/* The corrected style tag is placed here */}
            <style>{`
                html, body {
                    margin: 0;
                    padding: 0;
                    font-family: 'Inter', sans-serif;
                }
                .custom-scrollbar::-webkit-scrollbar {
                    width: 8px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: #1f2937;
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background-color: #4b5563;
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background-color: #6b7280;
                }
            `}</style>
        </div>
    );
};

export default App;
