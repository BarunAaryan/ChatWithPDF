import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Send } from 'lucide-react';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Card, CardContent, CardFooter, CardHeader } from './components/ui/card';

const API_URL = process.env.REACT_APP_API_URL;

function App() {
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [userId, setUserId] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setIsLoading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);
      try {
        const response = await axios.post(`${API_URL}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        console.log('Upload response:', response.data);
        setUserId(response.data.user_id);
        setMessages([{ id: Date.now().toString(), content: 'PDF uploaded successfully. You can now ask questions about its content.', sender: 'ai' }]);
      } catch (error) {
        console.error('Error uploading file:', error.response ? error.response.data : error.message);
        setMessages([{ id: Date.now().toString(), content: `Error uploading PDF: ${error.response ? error.response.data.detail : error.message}. Please try again.`, sender: 'ai' }]);
      }
      setIsLoading(false);
    }
  };

  //open file selection dialog box
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || !file) return;

    const userMessage = { id: Date.now().toString(), content: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_URL}/ask`, { question: input }, {
        params: { user_id: userId }
      });
      const aiMessage = { id: (Date.now() + 1).toString(), content: response.data.answer, sender: 'ai' };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error asking question:', error.response ? error.response.data : error.message);
      const errorMessage = { id: (Date.now() + 1).toString(), content: `Error processing your question: ${error.response ? error.response.data.detail : error.message}. Please try again.`, sender: 'ai' };
      setMessages(prev => [...prev, errorMessage]);
    }

    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <Card className="mx-auto max-w-[1200px] h-[90vh] flex flex-col">
        <CardHeader className="flex flex-row items-center justify-between border-b p-6">
          <img 
    src="src\components\images\7833886cropped.jpg" // logo image
    alt="PDF Chat Logo"
    className="h-12 w-auto"
  />
          <Button 
            onClick={handleUploadClick}
            variant="outline" 
            className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 h-12 text-lg"
          >
            <Upload className="w-5 h-5 mr-2" />
            Upload PDF
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={handleFileUpload}
          />
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'ai' ? 'justify-start' : 'justify-end'}`}
            >
              <div className={`max-w-[80%] rounded-lg p-4 ${
                message.sender === 'ai' ? 'bg-gray-100' : 'bg-green-500 text-white'
              }`}>
                <p className="text-base">{message.content}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="text-center text-gray-500 text-lg">Loading...</div>
          )}
        </CardContent>

        <CardFooter className="border-t p-6">
          <form onSubmit={handleSendMessage} className="flex gap-4 w-full">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the PDF..."
              disabled={!file || isLoading}
              className="flex-grow text-base h-12"
            />
            <Button 
              type="submit" 
              disabled={!file || isLoading} 
              className="bg-green-500 hover:bg-green-600 text-white px-6 h-12 text-lg"
            >
              <Send className="w-5 h-5" />
              <span className="sr-only">Send message</span>
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  );
}

export default App;