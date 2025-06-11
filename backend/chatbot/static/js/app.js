/**
 * YuhHearDem Chat Application
 * Real-time parliamentary research assistant with multi-agent streaming
 */

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0;
        var v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

var chatApp = {
    // Configuration
    apiBase: window.location.origin,
    userId: null,
    sessionId: null,
    isProcessing: false,
    currentStatusPanel: null,
    currentAssistantMessage: null, // Track current assistant message for replacement
    
    // DOM elements cache
    elements: {
        chatContainer: null,
        queryInput: null,
        sendButton: null,
        inputStatus: null,
        connectionStatus: null,
        clearButton: null
    },
    
    /**
     * Initialize the chat application
     */
    init: function() {
        console.log('🔧 Initializing YuhHearDem chat...');
        
        this.initializeSession();
        this.cacheElements();
        this.displaySessionInfo();
        this.testConnection();
        this.setupEventListeners();
    },
    
    /**
     * Cache DOM elements for performance
     */
    cacheElements: function() {
        this.elements.chatContainer = document.getElementById('chatContainer');
        this.elements.queryInput = document.getElementById('queryInput');
        this.elements.sendButton = document.getElementById('sendButton');
        this.elements.inputStatus = document.getElementById('inputStatus');
        this.elements.connectionStatus = document.getElementById('connectionStatus');
        this.elements.clearButton = document.getElementById('clearChat');
    },
    
    /**
     * Initialize or restore user session
     */
    initializeSession: function() {
        var existingUserId = sessionStorage.getItem('yuhheardem_user_id');
        var existingSessionId = sessionStorage.getItem('yuhheardem_session_id');
        
        if (existingUserId && existingSessionId) {
            console.log('🔄 Restoring existing session:', existingSessionId.substring(0, 8));
            this.userId = existingUserId;
            this.sessionId = existingSessionId;
            this.setSessionStatus('Session Restored', 'success');
        } else {
            console.log('🆕 Creating new session');
            this.userId = generateUUID();
            this.sessionId = generateUUID();
            
            sessionStorage.setItem('yuhheardem_user_id', this.userId);
            sessionStorage.setItem('yuhheardem_session_id', this.sessionId);
            this.setSessionStatus('New Session', 'new');
        }
        
        console.log('Session initialized - User:', this.userId.substring(0, 8), 'Session:', this.sessionId.substring(0, 8));
    },
    
    /**
     * Display session information in the UI
     */
    displaySessionInfo: function() {
        document.getElementById('sessionId').textContent = this.sessionId.substring(0, 8) + '...';
        document.getElementById('userId').textContent = this.userId.substring(0, 8) + '...';
    },
    
    /**
     * Set session status indicator
     */
    setSessionStatus: function(text, type) {
        var sessionStatus = document.getElementById('sessionStatus');
        sessionStatus.textContent = text;
        
        var className = 'ml-2 px-2 py-1 rounded text-xs ';
        switch(type) {
            case 'success':
                className += 'bg-green-100 text-green-800';
                break;
            case 'new':
                className += 'bg-blue-100 text-blue-800';
                break;
            case 'error':
                className += 'bg-red-100 text-red-800';
                break;
            default:
                className += 'bg-gray-100 text-gray-800';
        }
        sessionStatus.className = className;
    },
    
    /**
     * Test connection to the backend service
     */
    testConnection: function() {
        console.log('🔍 Testing connection to: ' + this.apiBase + '/health');
        
        fetch(this.apiBase + '/health')
            .then(function(response) {
                console.log('📡 Health response status: ' + response.status);
                if (!response.ok) {
                    throw new Error('Health check failed: ' + response.status);
                }
                return response.json();
            })
            .then(function(health) {
                console.log('📊 Health data:', health);
                if (health.status === 'healthy') {
                    chatApp.setConnectionStatus(true);
                    chatApp.enableInput();
                    chatApp.elements.inputStatus.textContent = 'Ready! Ask me about parliamentary discussions.';
                    console.log('✅ Connection successful');
                } else {
                    throw new Error('Service not healthy: ' + health.status);
                }
            })
            .catch(function(error) {
                console.error('❌ Connection failed:', error);
                chatApp.setConnectionStatus(false);
                chatApp.elements.inputStatus.textContent = 'Connection failed: ' + error.message;
            });
    },
    
    /**
     * Setup event listeners for UI interactions
     */
    setupEventListeners: function() {
        var self = this;
        
        this.elements.sendButton.addEventListener('click', function() {
            self.sendQuery();
        });
        
        this.elements.queryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                self.sendQuery();
            }
        });
        
        this.elements.clearButton.addEventListener('click', function() {
            self.clearChat();
        });
        
        // Auto-resize input based on content
        this.elements.queryInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    },
    
    /**
     * Update connection status indicator
     */
    setConnectionStatus: function(connected) {
        var statusDot = this.elements.connectionStatus.querySelector('div');
        var statusText = this.elements.connectionStatus.querySelector('span');
        
        if (connected) {
            statusDot.className = 'w-3 h-3 rounded-full bg-green-400 mr-2 connection-status-connected';
            statusText.textContent = 'Connected';
        } else {
            statusDot.className = 'w-3 h-3 rounded-full bg-red-400 mr-2 connection-status-disconnected';
            statusText.textContent = 'Disconnected';
        }
    },
    
    /**
     * Enable user input controls
     */
    enableInput: function() {
        this.elements.queryInput.disabled = false;
        this.elements.sendButton.disabled = false;
        this.elements.queryInput.focus();
    },
    
    /**
     * Disable user input controls
     */
    disableInput: function() {
        this.elements.queryInput.disabled = true;
        this.elements.sendButton.disabled = true;
    },
    
    /**
     * Send user query to the backend
     */
    sendQuery: function() {
        var query = this.elements.queryInput.value.trim();
        if (!query || this.isProcessing) return;
        
        this.isProcessing = true;
        this.disableInput();
        
        var userMessageDiv = this.addMessage('user', query);
        this.elements.queryInput.value = '';
        this.elements.queryInput.style.height = 'auto';
        
        // Reset tracking variables
        this.currentAssistantMessage = null;
        this.currentStatusPanel = null;
        
        this.streamQuery(query);
    },
    
    /**
     * Create a status panel underneath a message
     */
    createStatusPanel: function(messageDiv) {
        var statusPanel = document.createElement('div');
        statusPanel.className = 'status-panel mt-2 ml-11';
        statusPanel.innerHTML = 
            '<div class="bg-gray-50 border border-gray-200 rounded-lg p-3">' +
                '<div class="flex items-center space-x-2 mb-2">' +
                    '<div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>' +
                    '<span class="text-sm font-medium text-gray-700" id="currentStatus">Processing your question...</span>' +
                '</div>' +
                '<div class="hidden" id="statusDetails">' +
                    '<div class="text-xs text-gray-600 space-y-1" id="statusLog"></div>' +
                '</div>' +
            '</div>';
        
        messageDiv.appendChild(statusPanel);
        this.scrollToBottom();
        
        return statusPanel;
    },
    
    /**
     * Update current status in the panel
     */
    updateCurrentStatus: function(agent, message) {
        if (!this.currentStatusPanel) {
            console.log('❌ No status panel to update');
            return;
        }
        
        var statusElement = this.currentStatusPanel.querySelector('#currentStatus');
        if (statusElement) {
            console.log('✅ Updating status element:', message);
            statusElement.textContent = message;
        } else {
            console.log('❌ Could not find #currentStatus element in panel');
            console.log('Panel HTML:', this.currentStatusPanel.innerHTML);
        }
        
        // Add to detailed log
        this.addToStatusLog(agent, message);
    },
    
    /**
     * Add entry to status log
     */
    addToStatusLog: function(agent, message) {
        if (!this.currentStatusPanel) return;
        
        var statusLog = this.currentStatusPanel.querySelector('#statusLog');
        if (statusLog) {
            var timestamp = new Date().toLocaleTimeString();
            var logEntry = document.createElement('div');
            logEntry.className = 'text-xs text-gray-500';
            logEntry.innerHTML = 
                '<span class="text-gray-400">[' + timestamp + ']</span> ' +
                '<span class="font-medium text-blue-600">' + agent + ':</span> ' +
                '<span>' + message + '</span>';
            
            statusLog.appendChild(logEntry);
        }
    },
    
    /**
     * Collapse status panel into expandable section
     */
    collapseStatusPanel: function() {
        if (!this.currentStatusPanel) return;
        
        var statusElement = this.currentStatusPanel.querySelector('#currentStatus');
        var statusDetails = this.currentStatusPanel.querySelector('#statusDetails');
        var statusLog = this.currentStatusPanel.querySelector('#statusLog');
        
        if (statusElement && statusDetails && statusLog) {
            // Update main status
            statusElement.textContent = 'Processing completed';
            statusElement.className = 'text-sm text-gray-600 cursor-pointer hover:text-blue-600';
            
            // Remove pulsing dot and add expand icon
            var pulsingDot = this.currentStatusPanel.querySelector('.animate-pulse');
            if (pulsingDot) {
                pulsingDot.className = 'w-2 h-2 bg-green-500 rounded-full';
            }
            
            // Add expand/collapse functionality
            var toggleButton = document.createElement('button');
            toggleButton.className = 'ml-2 text-xs text-blue-500 hover:text-blue-700';
            toggleButton.textContent = 'Show details';
            toggleButton.onclick = function() {
                var isHidden = statusDetails.classList.contains('hidden');
                if (isHidden) {
                    statusDetails.classList.remove('hidden');
                    toggleButton.textContent = 'Hide details';
                } else {
                    statusDetails.classList.add('hidden');
                    toggleButton.textContent = 'Show details';
                }
            };
            
            statusElement.parentNode.appendChild(toggleButton);
            
            // Show that details are available
            if (statusLog.children.length > 0) {
                statusDetails.classList.remove('hidden');
                setTimeout(function() {
                    statusDetails.classList.add('hidden');
                    toggleButton.textContent = 'Show details';
                }, 3000); // Auto-collapse after 3 seconds
            }
        }
        
        this.currentStatusPanel = null;
    },
    
    /**
     * Stream query with real-time updates
     */
    streamQuery: function(query) {
        var self = this;
        
        fetch(this.apiBase + '/query/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                user_id: this.userId,
                session_id: this.sessionId
            })
        })
        .then(function(response) {
            if (!response.ok) {
                throw new Error('HTTP ' + response.status + ': ' + response.statusText);
            }
            return response.body.getReader();
        })
        .then(function(reader) {
            var decoder = new TextDecoder();
            var buffer = '';
            
            function readStream() {
                return reader.read().then(function(result) {
                    if (result.done) {
                        self.collapseStatusPanel();
                        self.isProcessing = false;
                        self.enableInput();
                        return;
                    }
                    
                    buffer += decoder.decode(result.value, { stream: true });
                    var lines = buffer.split('\n');
                    buffer = lines.pop();
                    
                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i];
                        
                        if (line.startsWith('data: ')) {
                            try {
                                var eventData = JSON.parse(line.substring(6));
                                console.log('📡 Real-time event:', eventData);
                                
                                // Safety check - ensure eventData exists
                                if (!eventData || !eventData.type) {
                                    console.log('⚠️ Invalid event data, skipping');
                                    continue;
                                }
                                
                                console.log('🔍 Event type check:', eventData.type, 'Panel exists:', !!self.currentStatusPanel);
                                
                                // Handle events in strict order - immediate_response first
                                if (eventData.type === 'immediate_response' && eventData.data && eventData.data.response) {
                                    console.log('💬 Immediate response received:', eventData.data.response);
                                    // Add the immediate conversational response
                                    self.currentAssistantMessage = self.addMessage('assistant', eventData.data.response);
                                    // Mark this message as conversational for potential replacement
                                    self.currentAssistantMessage.setAttribute('data-response-type', 'conversational');
                                    
                                    // NOW create the status panel under the assistant's response
                                    self.currentStatusPanel = self.createStatusPanel(self.currentAssistantMessage);
                                    console.log('🎛️ Status panel created:', !!self.currentStatusPanel);
                                    
                                } else if (eventData.type === 'final_response' && eventData.data && eventData.data.response) {
                                    console.log('📝 Final response received:', eventData.data.response.substring(0, 100) + '...');
                                    
                                    // If we have a current conversational message, replace it with the detailed response
                                    if (self.currentAssistantMessage && 
                                        self.currentAssistantMessage.getAttribute('data-response-type') === 'conversational') {
                                        self.replaceMessageContent(self.currentAssistantMessage, eventData.data.response);
                                        self.currentAssistantMessage.setAttribute('data-response-type', 'research');
                                    } else {
                                        // Otherwise, add as a new message
                                        self.currentAssistantMessage = self.addMessage('assistant', eventData.data.response);
                                        self.currentAssistantMessage.setAttribute('data-response-type', 'research');
                                    }
                                    
                                } else if (eventData.type === 'error') {
                                    console.log('❌ Error event:', eventData.message);
                                    self.addMessage('assistant', 'Error: ' + eventData.message);
                                    
                                } else if (eventData.type === 'stream_complete') {
                                    // Don't show stream_complete as a status update
                                    console.log('🏁 Stream completed');
                                    
                                } else {
                                    console.log('🔄 Processing other event type:', eventData.type);
                                    // For all other event types - update status if we have a panel
                                    if (self.currentStatusPanel && eventData.message) {
                                        console.log('📝 Updating status:', eventData.type, eventData.message);
                                        self.updateCurrentStatus(eventData.agent || 'System', eventData.message);
                                    } else {
                                        console.log('⏸️ Skipping status update:', {
                                            type: eventData.type,
                                            message: eventData.message,
                                            hasPanel: !!self.currentStatusPanel,
                                            hasMessage: !!eventData.message
                                        });
                                    }
                                }
                            } catch (e) {
                                console.error('❌ Error parsing SSE event:', e, 'Line:', line);
                            }
                        }
                    }
                    
                    return readStream();
                });
            }
            
            return readStream();
        })
        .catch(function(error) {
            console.error('Query failed:', error);
            self.addMessage('assistant', 'Sorry, I encountered an error: ' + error.message);
            self.collapseStatusPanel();
            self.isProcessing = false;
            self.enableInput();
        });
    },
    
    /**
     * Replace the content of an existing message
     */
    replaceMessageContent: function(messageDiv, newContent) {
        var contentElement = messageDiv.querySelector('.text-gray-800');
        if (contentElement) {
            // Use the content directly as HTML (already processed by backend)
            contentElement.innerHTML = newContent;
            
            // Update timestamp
            var timestampElement = messageDiv.querySelector('.text-xs.text-gray-500');
            if (timestampElement) {
                timestampElement.textContent = new Date().toLocaleTimeString();
            }
            
            this.scrollToBottom();
        }
    },
    
    /**
     * Add message to chat
     */
    addMessage: function(sender, content) {
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message-bubble';
        
        var isUser = sender === 'user';
        var avatarBg = isUser ? 'bg-green-500' : 'bg-blue-500';
        var bubbleBg = isUser ? 'bg-green-50' : 'bg-blue-50';
        var avatarText = isUser ? 'U' : 'YH';
        
        // Process content differently based on sender
        var htmlContent;
        if (isUser) {
            // For user messages, escape HTML and do basic text processing
            htmlContent = this.escapeHtml(content).replace(/\n/g, '<br>');
        } else {
            // For assistant messages, use content as-is (already processed HTML from backend)
            htmlContent = content;
        }
        
        var reverseClass = isUser ? 'flex-row-reverse space-x-reverse' : '';
        
        messageDiv.innerHTML = 
            '<div class="flex items-start space-x-3 ' + reverseClass + '">' +
                '<div class="w-8 h-8 rounded-full ' + avatarBg + ' flex items-center justify-center text-white text-sm font-semibold">' + avatarText + '</div>' +
                '<div class="flex-1">' +
                    '<div class="' + bubbleBg + ' rounded-lg p-4">' +
                        '<div class="text-gray-800">' + htmlContent + '</div>' +
                        '<div class="text-xs text-gray-500 mt-2">' + new Date().toLocaleTimeString() + '</div>' +
                    '</div>' +
                '</div>' +
            '</div>';
        
        this.elements.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    },
    
    /**
     * Escape HTML entities for user input
     */
    escapeHtml: function(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    /**
     * Scroll chat to bottom
     */
    scrollToBottom: function() {
        this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
    },
    
    /**
     * Clear chat and start new session
     */
    clearChat: function() {
        // Remove all messages except the welcome message
        var messages = this.elements.chatContainer.querySelectorAll('.message-bubble');
        for (var i = 1; i < messages.length; i++) {
            messages[i].remove();
        }
        
        // Generate new session
        this.userId = generateUUID();
        this.sessionId = generateUUID();
        
        sessionStorage.setItem('yuhheardem_user_id', this.userId);
        sessionStorage.setItem('yuhheardem_session_id', this.sessionId);
        
        this.displaySessionInfo();
        this.setSessionStatus('New Session', 'new');
        this.currentStatusPanel = null;
        this.currentAssistantMessage = null;
        
        console.log('🧹 Cleared chat and created new session:', this.sessionId.substring(0, 8));
        
        // Clear backend session
        this.clearBackendSession();
        
        this.elements.queryInput.focus();
    },
    
    /**
     * Clear backend session state
     */
    clearBackendSession: function() {
        fetch(this.apiBase + '/session/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: this.userId,
                session_id: this.sessionId
            })
        }).catch(function(error) {
            console.log('Backend session clear failed (non-critical):', error);
        });
    }
};

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    chatApp.init();
});

// Export for global access if needed
window.chatApp = chatApp;