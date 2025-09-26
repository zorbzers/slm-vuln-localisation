# File:         prompts/prompt_builder.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Prompt builder for generating zero-shot, one-shot and 
#               three-shot prompts to identify vulnerable lines in code 
#               snippets.

class PromptBuilder:
    """Class to build prompts for code vulnerability identification."""
    def __init__(self, shots: int = 0):
        """
        Initializes the PromptBuilder with a template for generating prompts.
        
        Parameters
        ----------
        shots : int
            Number of example shots to include in the prompt (0, 1, or 3).
        """
        if shots not in (0, 1, 3):
            raise ValueError("shots must be 0, 1, or 3")
        self.shots = shots
        
        self.examples = [
            {
                "code": (
                    "void Verify_StoreExistingGroup() {\n"
                    "    EXPECT_TRUE(delegate()->stored_group_success_);\n"
                    "    EXPECT_EQ(group_.get(), delegate()->stored_group_.get());\n"
                    "    EXPECT_EQ(cache2_.get(), group_->newest_complete_cache());\n"
                    "    EXPECT_TRUE(cache2_->is_complete());\n"
                    "\n"
                    "    AppCacheDatabase::GroupRecord group_record;\n"
                    "    AppCacheDatabase::CacheRecord cache_record;\n"
                    "    EXPECT_TRUE(database()->FindGroup(1, &group_record));\n"
                    "    EXPECT_TRUE(database()->FindCache(2, &cache_record));\n"
                    "\n"
                    "    EXPECT_FALSE(database()->FindCache(1, &cache_record));\n"
                    "\n"
                    "    EXPECT_EQ(kDefaultEntrySize + 100, storage()->usage_map_[kOrigin]);\n"
                    "    EXPECT_EQ(1, mock_quota_manager_proxy_->notify_storage_modified_count_);\n"
                    "    EXPECT_EQ(kOrigin, mock_quota_manager_proxy_->last_origin_);\n"
                    "    EXPECT_EQ(100, mock_quota_manager_proxy_->last_delta_);\n"
                    "\n"
                    "    TestFinished();\n"
                    "}\n"
                ),
                "vulnerable_lines": "[14,17]"
            },
            {
                "code": (
                    "void RenderViewImpl::NavigateBackForwardSoon(int offset) {\n"
                    "  history_navigation_virtual_time_pauser_ =\n"
                    "      RenderThreadImpl::current()\n"
                    "          ->GetWebMainThreadScheduler()\n"
                    "          ->CreateWebScopedVirtualTimePauser(\n"
                    "              \"NavigateBackForwardSoon\",\n"
                    "              blink::WebScopedVirtualTimePauser::VirtualTaskDuration::kInstant);\n"
                    "  history_navigation_virtual_time_pauser_.PauseVirtualTime();\n"
                    "  Send(new ViewHostMsg_GoToEntryAtOffset(GetRoutingID(), offset));\n"
                    "}\n"
                ),
                "vulnerable_lines": "[9]"
            },
            {
                "code": (
                    "base::string16 GetAppForProtocolUsingAssocQuery(const GURL& url) {\n"
                    "  base::string16 url_scheme = base::ASCIIToUTF16(url.scheme());\n"
                    "  if (url_scheme.empty())\n"
                    "     return base::string16();\n"
                    "\n"
                    "  wchar_t out_buffer[1024];\n"
                    "  DWORD buffer_size = arraysize(out_buffer);\n"
                    "  HRESULT hr = AssocQueryString(ASSOCF_IS_PROTOCOL,\n"
                    "                                ASSOCSTR_FRIENDLYAPPNAME,\n"
                    "                                url_scheme.c_str(),\n"
                    "                                NULL,\n"
                    "                                out_buffer,\n"
                    "                                &buffer_size);\n"
                    "  if (FAILED(hr)) {\n"
                    "    DLOG(WARNING) << \"AssocQueryString failed!\";\n"
                    "    return base::string16();\n"
                    "  }\n"
                    "  return base::string16(out_buffer);\n"
                    "}\n"
                ),
                "vulnerable_lines": "[2,4,9,10,11,12,13,14]"
            }
        ]
        
        self.base_prompt = (
            "Given the following code snippet, identify the vulnerable line(s). Vulnerable lines are those that require modification for the code to be secure.\n"
            "Code:\n{code}\n\n"
            "Respond strictly with a comma separated list of vulnerable line numbers, enclosed in square brackets.\n"
            "Do not explain your reasoning. "
            "Do not prefix your response with any text. "
            "Do not include any whitespace. "
            "Do not include any additional text or explanations.\n\n"
        )
        
        
    def _format_examples(self) -> str:
        """
        Formats the example shots for inclusion in the prompt.
        
        Returns
        -------
        str
            Formatted example shots string.
        """
        if self.shots == 0:
            return ""
        
        if self.shots == 1:
            label = "Example:\n"
        else:
            label = "Examples:\n"
            
        examples_str = "".join(
            f"Code:\n{ex['code']}\nVulnerable lines: {ex['vulnerable_lines']}\n\n"
            for ex in self.examples[:self.shots]
        )
        
        return label + examples_str
  
        
    def build_base_prompt(self, code: str) -> str:
        """
        Builds a base prompt using the provided code snippet.
        
        Parameters
        ----------
        code : str
            The code snippet to be analyzed.
            
        Returns
        -------
        str
            A formatted prompt string.
        """
        return self._format_examples() + self.base_prompt.format(code=code)


    def build_highlighted_prompt(self, code: str, highlight_line: str) -> str:
        """
        Builds a prompt with a highlighted line to draw attention to a specific part of the code.
        
        Parameters
        ----------
        code : str
            The code snippet to be analyzed.
        highlight_line : str
            The line number to highlight in the prompt.
            
        Returns
        -------
        str
            A formatted prompt string with the highlighted line.
        """
        return self._format_examples() + self.base_prompt.format(code=code) + f"Pay attention to line {highlight_line}.\n"
    
    
    def build_all_highlighted_prompts(self, code: str, valid_lines: list[int]) -> list:
        """
        Builds a list of prompts for each line in the code snippet, highlighting each line one by one.
        
        Parameters
        ----------
        code : str
            The code snippet to be analyzed.
        valid_lines : list of int
            List of valid line numbers in the code snippet.
            
        Returns
        -------
        list[str]
            A list of formatted prompt strings, each with a different line highlighted.
        """ 
        return [self.build_highlighted_prompt(code, str(i)) for i in valid_lines]