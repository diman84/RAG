import * as vscode from 'vscode';
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { Document } from '@langchain/core/documents';
import { FaissStore }from "@langchain/community/vectorstores/faiss";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

export class Indexer {
    private workspacePath: string | undefined;
    private docs: Document[] = [];
    private embeddings: OpenAIEmbeddings;
    private vectorStore: MemoryVectorStore;

    constructor() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
            this.workspacePath = workspaceFolders[0].uri.fsPath;
        } else {
            console.log('No workspace folder found');
        }

        this.embeddings = new OpenAIEmbeddings({
            model: "text-embedding-3-small",
          });
          
        this.vectorStore = new MemoryVectorStore(this.embeddings, {});
    }

    async indexRepositoryFiles(): Promise<void> {
        try {
            if (this.workspacePath) {
                await Promise.all([
                    this._loadJs(this.workspacePath),
                    this._loadMd(this.workspacePath)
                ]);

                await this.vectorStore.addDocuments(this.docs);
            } else {
                console.log('No workspace folder found');
            }
        } catch (error) {
            console.error('Error indexing repository files:', error);
        }
    }

    async searchRelatedDocuments(query: string, k: number): Promise<Document[]> {
        if (this.vectorStore.maxMarginalRelevanceSearch) {
            return await this.vectorStore.maxMarginalRelevanceSearch(query, {k: k});
        }
        
        return this.vectorStore.similaritySearch(query, k);
    }

    async _loadJs(path: string): Promise<void> {        
        const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
            chunkSize: 100,
            chunkOverlap: 0,
          });
        const loader = new DirectoryLoader(path, {
            ".js": (path) => new TextLoader(path),
        });

        var jsDocs = await loader.load();        
        this.docs.push(...await splitter.transformDocuments(jsDocs));
    }

    
    async _loadMd(path: string): Promise<void> {        
        const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
            chunkSize: 300,
            chunkOverlap: 10,
          });
        const loader = new DirectoryLoader(path, {
            ".md": (path) => new TextLoader(path),
        });

        var jsDocs = await loader.load();        
        this.docs.push(...await splitter.transformDocuments(jsDocs));
    }
}