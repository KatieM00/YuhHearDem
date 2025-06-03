import React from 'react';
import { useAppContext } from '../../context/AppContext';
import { Briefcase, Activity, Building2, BarChart3 } from 'lucide-react';

const StarterPrompts: React.FC = () => {
  const { setInput, sendMessage } = useAppContext();

  const handlePromptClick = (prompt: string) => {
    setInput(prompt);
    setTimeout(() => {
      sendMessage();
    }, 100);
  };

  const prompts = [
    {
      icon: <Briefcase className="mr-2" />,
      text: "What are the latest economic reforms?",
      gradient: "from-yellow-400 via-orange-400 to-red-400"
    },
    {
      icon: <Activity className="mr-2" />,
      text: "Updates on healthcare initiatives?",
      gradient: "from-teal-400 via-blue-400 to-indigo-400"
    },
    {
      icon: <Building2 className="mr-2" />,
      text: "Recent tourism policy changes?",
      gradient: "from-pink-400 via-purple-400 to-violet-400"
    },
    {
      icon: <BarChart3 className="mr-2" />,
      text: "Last parliamentary session summary?",
      gradient: "from-green-400 via-emerald-400 to-teal-400"
    }
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      <h3 className="text-lg font-medium text-gray-700 mb-4">
        Get started by asking about:
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {prompts.map((prompt, index) => (
          <button 
            key={index}
            onClick={() => handlePromptClick(prompt.text)}
            className={`bg-gradient-to-br ${prompt.gradient} p-5 rounded-2xl shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 text-white group`}
          >
            <div className="flex items-center">
              {prompt.icon}
              <span className="font-semibold">{prompt.text}</span>
            </div>
          </button>
        ))}
      </div>
      
      <div className="mt-8 text-center">
        <p className="text-gray-600 text-sm">
          Or type your own question below to learn about Barbadian politics
        </p>
      </div>
    </div>
  );
};

export default StarterPrompts;