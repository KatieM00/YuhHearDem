import React from 'react';
import { useAppContext } from '../context/AppContext';
import { X, ExternalLink } from 'lucide-react';

const AboutModal: React.FC = () => {
  const { showAboutModal, setShowAboutModal } = useAppContext();

  if (!showAboutModal) return null;

  const resources = [
    {
      title: "GOV.BB (Government of Barbados)",
      url: "https://www.gov.bb/",
      description: "Central hub for all government information, ministries, and official documents"
    },
    {
      title: "The Barbados Parliament",
      url: "https://www.barbadosparliament.com/",
      description: "Bills, resolutions, House of Assembly, Senate, and parliamentary structure"
    },
    {
      title: "Prime Minister's Office",
      url: "https://www.gov.bb/Ministries/prime-minister-office",
      description: "Details on the Prime Minister's office and responsibilities"
    },
    {
      title: "Government Forms",
      url: "https://forms.gov.bb/",
      description: "Official government processes and regulations"
    }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-3xl shadow-2xl border border-white/20 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-100">
          <div className="flex items-center space-x-3">
            <span className="text-2xl" role="img" aria-label="Barbados Flag">🇧🇧</span>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              About Yuh Hear Dem
            </h2>
          </div>
          <button
            onClick={() => setShowAboutModal(false)}
            className="w-10 h-10 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center transition-all duration-200"
            aria-label="Close modal"
          >
            <X size={20} className="text-gray-600" />
          </button>
        </div>

        {/* Main Content */}
        <div className="p-6 space-y-6">
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed text-lg mb-4">
              Ever wondered what your MPs are actually saying in Parliament? We get it - sifting through hours of parliamentary footage isn't exactly a Friday night activity! That's where we come in.
            </p>
            
            <p className="text-gray-700 leading-relaxed text-lg mb-4">
              Using cutting-edge AI, we've transformed those lengthy YouTube parliamentary sessions into your personal political assistant. Just ask a question like "What did they say about healthcare?" and boom - you get the answer with direct links to the exact moments in the videos where it was discussed.
            </p>
            
            <p className="text-gray-700 leading-relaxed text-lg mb-4">
              No more endless scrolling, no more political jargon confusion. Just straight answers from the source, timestamped and ready to watch. Whether you're a political junkie or just want to know what's happening in your community, we're here to make Barbadian politics as accessible as your morning coffee.
            </p>
            
            <p className="text-gray-700 leading-relaxed text-lg">
              Because let's face it - democracy works best when we're all in the know. And staying informed shouldn't feel like homework.
            </p>
          </div>

          {/* Additional Resources Section */}
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-gray-800 flex items-center">
              <span className="mr-2">📚</span>
              Additional Resources
            </h3>
            
            <div className="grid gap-4">
              {resources.map((resource, index) => (
                <a
                  key={index}
                  href={resource.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 border border-blue-200 hover:border-blue-300 rounded-2xl p-4 transition-all duration-300 transform hover:scale-[1.02] hover:shadow-lg"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-800 group-hover:text-blue-700 transition-colors duration-200 flex items-center">
                        {resource.title}
                        <ExternalLink size={16} className="ml-2 opacity-60 group-hover:opacity-100" />
                      </h4>
                      <p className="text-sm text-gray-600 mt-1 leading-relaxed">
                        {resource.description}
                      </p>
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="pt-4 border-t border-gray-100">
            <p className="text-center text-sm text-gray-500">
              Built with ❤️ for the people of Barbados
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutModal;