#include "server.h"
#include <iostream>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

InferenceServer::InferenceServer(std::shared_ptr<ExecutionEngine> engine, int port)
    : engine_(engine), port_(port) 
{
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

InferenceServer::~InferenceServer() {
#ifdef _WIN32
    WSACleanup();
#endif
}

void InferenceServer::start() {
    std::cout << "Starting Inference Server on port " << port_ << "..." << std::endl;
    // Dummy loop simulating a server for now
    // In a real implementation, we would bind, listen, and accept sockets here
    
    std::string dummy_payload = "{\"prompt\": \"Hello!\"}";
    std::cout << "Simulating incoming request: " << dummy_payload << std::endl;
    handle_client(dummy_payload);
    
    std::cout << "Server shutdown." << std::endl;
}

void InferenceServer::handle_client(const std::string& request_payload) {
    // Parse JSON, encode prompt, forward pass, decode tokens...
    std::vector<int> dummy_tokens = {1, 2, 3}; 
    engine_->reset();
    engine_->forward(dummy_tokens);
    std::cout << "Request processed successfully." << std::endl;
}
