window.neetcodeData = {
  "topics": [
    {
      "topic_name": "Arrays & Hashing",
      "problems": [
        {"id": 217, "title": "Contains Duplicate", "difficulty": "Easy", "link": "https://leetcode.com/problems/contains-duplicate/", "description": "Given an integer array `nums`, determine if any value appears at least twice in the array. Return `true` if a duplicate exists, otherwise return `false`.", "details": {"key_idea": "Use a hash set to store unique elements as we traverse the list. If an element is already present in the set, we have found a duplicate.", "time_complexity": "O(n)", "space_complexity": "O(n)", "python_solution": "class Solution:\n    def containsDuplicate(self, nums: List[int]) -> bool:\n        hashset = set()\n\n        for n in nums:\n            if n in hashset:\n                return True\n            hashset.add(n)\n        return False"}},
        {"id": 242, "title": "Valid Anagram", "difficulty": "Easy", "link": "https://leetcode.com/problems/valid-anagram/", "description": "Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise. An anagram is a word formed by rearranging the letters of another, using all original letters exactly once.", "details": {"key_idea": "Use a hash map to count character frequencies for each string. If the frequency maps are identical, they are anagrams.", "time_complexity": "O(n)", "space_complexity": "O(1)", "python_solution": "class Solution:\n    def isAnagram(self, s: str, t: str) -> bool:\n        if len(s) != len(t):\n            return False\n\n        char_frequency = {}\n\n        for char in s:\n            char_frequency[char] = char_frequency.get(char, 0) + 1\n\n        for char in t:\n            if char not in char_frequency or char_frequency[char] == 0:\n                return False\n            char_frequency[char] -= 1\n\n        return True"}},
        {"id": 1, "title": "Two Sum", "difficulty": "Easy", "link": "https://leetcode.com/problems/two-sum/", "description": "Given an array of integers `nums` and a `target`, return indices of the two numbers that add up to `target`. Assume exactly one solution exists and you may not use the same element twice.", "details": {"key_idea": "Use a hash map to store elements and their indices. For each element, calculate the required complement (`target` - `num`) and check if it's in the map.", "time_complexity": "O(n)", "space_complexity": "O(n)", "python_solution": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        prevMap = {}\n\n        for i, n in enumerate(nums):\n            diff = target - n\n            if diff in prevMap:\n                return [prevMap[diff], i]\n            prevMap[n] = i"}},
        {"id": 49, "title": "Group Anagrams", "difficulty": "Medium", "link": "https://leetcode.com/problems/group-anagrams/", "description": "Given an array of strings `strs`, group the anagrams together. You can return the answer in any order.", "details": {"key_idea": "Use a hash map where the key is a sorted version of each word (or a character count tuple). The value will be a list of its anagrams.", "time_complexity": "O(n * m log m)", "space_complexity": "O(n * m)", "python_solution": "class Solution:\n    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:\n        anagrams_map = {}\n\n        for word in strs:\n            sorted_word = \"\".join(sorted(word))\n            if sorted_word in anagrams_map:\n                anagrams_map[sorted_word].append(word)\n            else:\n                anagrams_map[sorted_word] = [word]\n\n        return list(anagrams_map.values())"}},
        {"id": 347, "title": "Top K Frequent Elements", "difficulty": "Medium", "link": "https://leetcode.com/problems/top-k-frequent-elements/", "description": "Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.", "details": {"key_idea": "Use a hash map for frequency counts. Then, use a min-heap of size `k` to keep track of the top `k` frequent elements, or use Bucket Sort for an O(n) solution.", "time_complexity": "O(n log k)", "space_complexity": "O(n)", "python_solution": "import heapq\n\nclass Solution:\n    def topKFrequent(self, nums: List[int], k: int) -> List[int]:\n        frequency_map = {}\n        for num in nums:\n            frequency_map[num] = frequency_map.get(num, 0) + 1\n\n        min_heap = []\n        for num, frequency in frequency_map.items():\n            heapq.heappush(min_heap, (frequency, num))\n            if len(min_heap) > k:\n                heapq.heappop(min_heap)\n\n        return [num for frequency, num in min_heap]"}},
        {"id": 238, "title": "Product of Array Except Self", "difficulty": "Medium", "link": "https://leetcode.com/problems/product-of-array-except-self/", "description": "Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`. You must write an algorithm that runs in O(n) time and without using the division operation.", "details": {"key_idea": "Calculate prefix products in one pass (left-to-right) and suffix products in a second pass (right-to-left). The result for each index is the product of its corresponding prefix and suffix.", "time_complexity": "O(n)", "space_complexity": "O(1) (excluding output array)", "python_solution": "class Solution:\n    def productExceptSelf(self, nums: List[int]) -> List[int]:\n        n = len(nums)\n        result = [1] * n\n\n        left_product = 1\n        for i in range(n):\n            result[i] *= left_product\n            left_product *= nums[i]\n\n        right_product = 1\n        for i in range(n - 1, -1, -1):\n            result[i] *= right_product\n            right_product *= nums[i]\n\n        return result"}}
      ]
    },
    {
        "topic_name": "Two Pointers",
        "problems": [
        {"id": 125, "title": "Valid Palindrome", "difficulty": "Easy", "link": "https://leetcode.com/problems/valid-palindrome/", "description": "Given a string `s`, determine if it is a palindrome after converting all uppercase letters to lowercase and removing all non-alphanumeric characters.", "details": {"key_idea": "Use two pointers, one at the start and one at the end. Move them inwards, skipping non-alphanumeric characters, and compare the characters case-insensitively.", "time_complexity": "O(n)", "space_complexity": "O(1)", "python_solution": "class Solution:\n    def isPalindrome(self, s: str) -> bool:\n        left, right = 0, len(s) - 1\n\n        while left < right:\n            while left < right and not s[left].isalnum():\n                left += 1\n            while left < right and not s[right].isalnum():\n                right -= 1\n\n            if s[left].lower() != s[right].lower():\n                return False\n\n            left += 1\n            right -= 1\n\n        return True"}},
        {"id": 167, "title": "Two Sum II - Input Array Is Sorted", "difficulty": "Medium", "link": "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/", "description": "Given a 1-indexed array of integers `numbers` that is already sorted in non-decreasing order, find two numbers such that they add up to a specific `target` number.", "details": {"key_idea": "Since the array is sorted, use two pointers (left at start, right at end). If sum is too small, move left pointer up. If too large, move right pointer down.", "time_complexity": "O(n)", "space_complexity": "O(1)", "python_solution": "class Solution:\n    def twoSum(self, numbers: List[int], target: int) -> List[int]:\n        left, right = 0, len(numbers) - 1\n\n        while left < right:\n            current_sum = numbers[left] + numbers[right]\n\n            if current_sum == target:\n                return [left + 1, right + 1]\n            elif current_sum < target:\n                left += 1\n            else:\n                right -= 1\n\n        return [-1, -1]"}},
        {"id": 15, "title": "3Sum", "difficulty": "Medium", "link": "https://leetcode.com/problems/3sum/", "description": "Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, `j != k`, and `nums[i] + nums[j] + nums[k] == 0`. The solution set must not contain duplicate triplets.", "details": {"key_idea": "Sort the array. Iterate with a main pointer `i`, then use two pointers (left and right) on the remainder of the array to find pairs that sum to -nums[i]. Skip duplicate elements to avoid duplicate triplets.", "time_complexity": "O(n^2)", "space_complexity": "O(1) or O(n) depending on sort", "python_solution": "class Solution:\n    def threeSum(self, nums: List[int]) -> List[List[int]]:\n        nums.sort()\n        result = []\n        n = len(nums)\n\n        for i in range(n - 2):\n            if i > 0 and nums[i] == nums[i - 1]:\n                continue\n\n            left, right = i + 1, n - 1\n\n            while left < right:\n                current_sum = nums[i] + nums[left] + nums[right]\n\n                if current_sum == 0:\n                    result.append([nums[i], nums[left], nums[right]])\n                    while left < right and nums[left] == nums[left + 1]:\n                        left += 1\n                    while left < right and nums[right] == nums[right - 1]:\n                        right -= 1\n                    left += 1\n                    right -= 1\n                elif current_sum < 0:\n                    left += 1\n                else:\n                    right -= 1\n\n        return result"}},
        {"id": 11, "title": "Container With Most Water", "difficulty": "Medium", "link": "https://leetcode.com/problems/container-with-most-water/", "description": "Given an integer array `height` of length `n`, find two lines that together with the x-axis form a container, such that the container contains the most water. Return the maximum amount of water a container can store.", "details": {"key_idea": "Use two pointers at the ends of the array. Calculate the area. Move the pointer with the shorter height inward, as this is the only way to potentially increase the area.", "time_complexity": "O(n)", "space_complexity": "O(1)", "python_solution": "class Solution:\n    def maxArea(self, height: List[int]) -> int:\n        left, right = 0, len(height) - 1\n        max_area = 0\n\n        while left < right:\n            current_area = min(height[left], height[right]) * (right - left)\n            max_area = max(max_area, current_area)\n\n            if height[left] < height[right]:\n                left += 1\n            else:\n                right -= 1\n\n        return max_area"}},
        {"id": 42, "title": "Trapping Rain Water", "difficulty": "Hard", "link": "https://leetcode.com/problems/trapping-rain-water/", "description": "Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.", "details": {"key_idea": "Use two pointers (left, right) and track max_left and max_right heights. The amount of water trapped at any point is determined by the minimum of max_left and max_right minus the current height.", "time_complexity": "O(n)", "space_complexity": "O(1)", "python_solution": "class Solution:\n    def trap(self, height: List[int]) -> int:\n        left, right = 0, len(height) - 1\n        max_left, max_right = 0, 0\n        trapped_water = 0\n\n        while left < right:\n            if height[left] <= height[right]:\n                if height[left] >= max_left:\n                    max_left = height[left]\n                else:\n                    trapped_water += max_left - height[left]\n                left += 1\n            else:\n                if height[right] >= max_right:\n                    max_right = height[right]\n                else:\n                    trapped_water += max_right - height[right]\n                right -= 1\n\n        return trapped_water"}}
        ]
    },
    {
        "topic_name": "Sliding Window",
        "problems": [
        {"id": 121, "title": "Best Time to Buy & Sell Stock", "difficulty": "Easy", "link": "https://leetcode.com/problems/best-time-to-buy-and-sell-stock/", "description": "You are given an array `prices` where `prices[i]` is the price of a given stock on the `i-th` day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve. If you cannot achieve any profit, return 0.", "details": {"key_idea": "Iterate through the prices, keeping track of the minimum price seen so far (min_price) and the maximum profit found (current_price - min_price).", "time_complexity": "O(n)", "space_complexity": "O(1)", "python_solution": "class Solution:\n    def maxProfit(self, prices: List[int]) -> int:\n        if not prices:\n            return 0\n\n        min_price = float(\"inf\")\n        max_profit = 0\n\n        for price in prices:\n            min_price = min(min_price, price)\n            max_profit = max(max_profit, price - min_price)\n\n        return max_profit"}},
        {"id": 3, "title": "Longest Substring Without Repeating Characters", "difficulty": "Medium", "link": "https://leetcode.com/problems/longest-substring-without-repeating-characters/", "description": "Given a string `s`, find the length of the longest substring without repeating characters.", "details": {"key_idea": "Use a sliding window with a set to track unique characters. Expand the window by moving the right pointer. If a duplicate is found, shrink the window from the left.", "time_complexity": "O(n)", "space_complexity": "O(k) where k is number of unique chars", "python_solution": "class Solution:\n    def lengthOfLongestSubstring(self, s: str) -> int:\n        left, right = 0, 0\n        max_length = 0\n        unique_chars = set()\n\n        while right < len(s):\n            if s[right] not in unique_chars:\n                unique_chars.add(s[right])\n                max_length = max(max_length, right - left + 1)\n                right += 1\n            else:\n                unique_chars.remove(s[left])\n                left += 1\n\n        return max_length"}},
        {"id": 424, "title": "Longest Repeating Character Replacement", "difficulty": "Medium", "link": "https://leetcode.com/problems/longest-repeating-character-replacement/", "description": "You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most `k` times. Return the length of the longest substring containing the same letter you can get after performing the above operations.", "details": {"key_idea": "The window is valid if (window_length - max_frequency) <= k. Use a sliding window and a frequency map. Expand the window, and if invalid, shrink from the left.", "time_complexity": "O(n)", "space_complexity": "O(1) (alphabet size is 26)", "python_solution": "class Solution:\n    def characterReplacement(self, s: str, k: int) -> int:\n        left, right = 0, 0\n        max_length = 0\n        char_freq = {}\n        max_freq = 0\n\n        while right < len(s):\n            char_freq[s[right]] = char_freq.get(s[right], 0) + 1\n            max_freq = max(max_freq, char_freq[s[right]])\n\n            if (right - left + 1) - max_freq > k:\n                char_freq[s[left]] -= 1\n                left += 1\n\n            max_length = max(max_length, right - left + 1)\n            right += 1\n\n        return max_length"}},
        {"id": 567, "title": "Permutation in String", "difficulty": "Medium", "link": "https://leetcode.com/problems/permutation-in-string/", "description": "Given two strings `s1` and `s2`, return `true` if `s2` contains a permutation of `s1`, or `false` otherwise. In other words, return `true` if one of `s1`'s permutations is the substring of `s2`.", "details": {"key_idea": "Use a sliding window of size len(s1) on s2. Maintain frequency maps for s1 and the current window. If maps are equal, a permutation exists.", "time_complexity": "O(n)", "space_complexity": "O(1)", "python_solution": "class Solution:\n    def checkInclusion(self, s1: str, s2: str) -> bool:\n        if len(s1) > len(s2):\n            return False\n\n        char_freq_s1 = {}\n        for char in s1:\n            char_freq_s1[char] = char_freq_s1.get(char, 0) + 1\n\n        left, right = 0, 0\n        char_freq_temp = {}\n\n        while right < len(s2):\n            char_freq_temp[s2[right]] = char_freq_temp.get(s2[right], 0) + 1\n\n            if right - left + 1 == len(s1):\n                if char_freq_temp == char_freq_s1:\n                    return True\n                char_freq_temp[s2[left]] -= 1\n                if char_freq_temp[s2[left]] == 0:\n                    del char_freq_temp[s2[left]]\n                left += 1\n\n            right += 1\n\n        return False"}},
        {"id": 76, "title": "Minimum Window Substring", "difficulty": "Hard", "link": "https://leetcode.com/problems/minimum-window-substring/", "description": "Given two strings `s` and `t`, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such substring, return an empty string `\"\"`.", "details": {"key_idea": "Use a sliding window. Expand the window until it contains all characters of t. Then, shrink the window from the left while it remains valid, updating the minimum length.", "time_complexity": "O(n)", "space_complexity": "O(k) where k is unique chars in t", "python_solution": "class Solution:\n    def minWindow(self, s: str, t: str) -> str:\n        if not s or not t:\n            return \"\"\n\n        char_freq_t = {}\n        for char in t:\n            char_freq_t[char] = char_freq_t.get(char, 0) + 1\n\n        left, right = 0, 0\n        char_freq_temp = {}\n        required_chars = len(char_freq_t)\n        formed_chars = 0\n        min_length = float(\"inf\")\n        min_window = \"\"\n\n        while right < len(s):\n            char_freq_temp[s[right]] = char_freq_temp.get(s[right], 0) + 1\n\n            if s[right] in char_freq_t and char_freq_temp[s[right]] == char_freq_t[s[right]]:\n                formed_chars += 1\n\n            while left <= right and formed_chars == required_chars:\n                if right - left + 1 < min_length:\n                    min_length = right - left + 1\n                    min_window = s[left : right + 1]\n\n                char_freq_temp[s[left]] -= 1\n                if s[left] in char_freq_t and char_freq_temp[s[left]] < char_freq_t[s[left]]:\n                    formed_chars -= 1\n\n                left += 1\n\n            right += 1\n\n        return min_window"}},
        {"id": 239, "title": "Sliding Window Maximum", "difficulty": "Hard", "link": "https://leetcode.com/problems/sliding-window-maximum/", "description": "You are given an array of integers `nums`, and a sliding window of size `k` moving from the very left to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.", "details": {"key_idea": "Use a deque to store indices. The deque is kept in decreasing order of element values. The front of the deque is always the max for the current window.", "time_complexity": "O(n)", "space_complexity": "O(k)", "python_solution": "from collections import deque\n\nclass Solution:\n    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:\n        if not nums or k <= 0:\n            return []\n\n        result = []\n        window = deque()\n\n        for i, num in enumerate(nums):\n            while window and nums[window[-1]] < num:\n                window.pop()\n\n            window.append(i)\n\n            if i - window[0] >= k:\n                window.popleft()\n\n            if i >= k - 1:\n                result.append(nums[window[0]])\n\n        return result"}}
        ]
    },
    {
      "topic_name": "Arrays & Hashing",
      "problems": [
        {
          "id": 217,
          "title": "Contains Duplicate",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/contains-duplicate/",
          "description": "Given an integer array nums, determine if any value appears at least twice in the array. Return true if a duplicate exists, otherwise return false.",
          "details": {
            "key_idea": "Use a hash set to store unique elements as we traverse the list. If an element is already present in the set, we have found a duplicate.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def containsDuplicate(self, nums: List[int]) -> bool:\n        hashset = set()\n\n        for n in nums:\n            if n in hashset:\n                return True\n            hashset.add(n)\n        return False"
          }
        },
        {
          "id": 242,
          "title": "Valid Anagram",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/valid-anagram/",
          "description": "Given two strings s and t, return true if t is an anagram of s, and false otherwise.",
          "details": {
            "key_idea": "To determine if two given strings are anagrams of each other, we can compare their character frequencies. An anagram of a string contains the same characters with the same frequency, just arranged differently. We can use a hash map (dictionary in Python) to keep track of the character frequencies for each string. If the character frequencies of both strings match, then they are anagrams.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def isAnagram(self, s: str, t: str) -> bool:\n        if len(s) != len(t):\n            return False\n\n        char_frequency = {}\n\n        # Build character frequency map for string s\n        for char in s:\n            char_frequency[char] = char_frequency.get(char, 0) + 1\n\n        # Compare with string t\n        for char in t:\n            if char not in char_frequency or char_frequency[char] == 0:\n                return False\n            char_frequency[char] -= 1\n\n        return True"
          }
        },
        {
          "id": 1,
          "title": "Two Sum",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/two-sum/",
          "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
          "details": {
            "key_idea": "The key idea to solve this problem efficiently is by using a hash map (dictionary in Python) to keep track of the elements we have traversed so far. For each element in the input list, we calculate the difference between the target and the current element. If this difference exists in the hash map, then we have found the pair that sums to the target, and we return their indices. Otherwise, we add the current element to the hash map and continue with the next element.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        prevMap = {}  # val -> index\n\n        for i, n in enumerate(nums):\n            diff = target - n\n            if diff in prevMap:\n                return [prevMap[diff], i]\n            prevMap[n] = i"
          }
        },
        {
          "id": 49,
          "title": "Group Anagrams",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/group-anagrams/",
          "description": "Given an array of strings strs, group the anagrams together. You can return the answer in any order.",
          "details": {
            "key_idea": "To group anagrams together, we can use a hash map (dictionary in Python) where the key is a sorted version of each word, and the value is a list of words that are anagrams of each other. By iterating through the list of words, we can group them into the hash map based on their sorted versions.",
            "time_complexity": "O(n * m * log(m))",
            "space_complexity": "O(n * m)",
            "python_solution": "class Solution:\n    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:\n        anagrams_map = {}\n\n        for word in strs:\n            sorted_word = \"\".join(sorted(word))\n            if sorted_word in anagrams_map:\n                anagrams_map[sorted_word].append(word)\n            else:\n                anagrams_map[sorted_word] = [word]\n\n        return list(anagrams_map.values())"
          }
        },
        {
          "id": 347,
          "title": "Top K Frequent Elements",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/top-k-frequent-elements/",
          "description": "Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.",
          "details": {
            "key_idea": "To find the k most frequent elements, we can use a hash map (dictionary in Python) to keep track of the frequency of each element. We then use a min-heap (priority queue) to keep the k most frequent elements at the top. We traverse the list once to build the frequency map, and then we traverse the map to keep the k most frequent elements in the min-heap.",
            "time_complexity": "O(n + k*log(n))",
            "space_complexity": "O(n)",
            "python_solution": "import heapq\n\n\nclass Solution:\n    def topKFrequent(self, nums: List[int], k: int) -> List[int]:\n        frequency_map = {}\n        for num in nums:\n            frequency_map[num] = frequency_map.get(num, 0) + 1\n\n        min_heap = []\n        for num, frequency in frequency_map.items():\n            heapq.heappush(min_heap, (frequency, num))\n            if len(min_heap) > k:\n                heapq.heappop(min_heap)\n\n        return [num for frequency, num in min_heap]"
          }
        },
        {
          "id": 238,
          "title": "Product of Array Except Self",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/product-of-array-except-self/",
          "description": "Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]. The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.",
          "details": {
            "key_idea": "To solve this problem, we can first calculate the product of all elements to the left of each index and store it in a list. Then, we calculate the product of all elements to the right of each index and update the result list accordingly by multiplying it with the previously calculated left product. In this way, each element in the result list will contain the product of all elements except the one at that index.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def productExceptSelf(self, nums: List[int]) -> List[int]:\n        n = len(nums)\n        result = [1] * n\n\n        # Calculate the left product of each element\n        left_product = 1\n        for i in range(n):\n            result[i] *= left_product\n            left_product *= nums[i]\n\n        # Calculate the right product of each element and update the result list\n        right_product = 1\n        for i in range(n - 1, -1, -1):\n            result[i] *= right_product\n            right_product *= nums[i]\n\n        return result"
          }
        },
        {
          "id": 36,
          "title": "Valid Sudoku",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/valid-sudoku/",
          "description": "Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules: Each row must contain the digits 1-9 without repetition. Each column must contain the digits 1-9 without repetition. Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.",
          "details": {
            "key_idea": "To determine if a given Sudoku board is valid, we need to check three conditions: 1. Each row must have distinct digits from 1 to 9. 2. Each column must have distinct digits from 1 to 9. 3. Each 3x3 sub-grid must have distinct digits from 1 to 9. We can use three nested loops to traverse the entire board and use sets to keep track of digits seen in each row, column, and sub-grid.",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def isValidSudoku(self, board: List[List[str]]) -> bool:\n        seen = set()\n\n        for i in range(9):\n            for j in range(9):\n                if board[i][j] != \".\":\n                    num = board[i][j]\n                    if (\n                        (i, num) in seen\n                        or (num, j) in seen\n                        or (i // 3, j // 3, num) in seen\n                    ):\n                        return False\n                    seen.add((i, num))\n                    seen.add((num, j))\n                    seen.add((i // 3, j // 3, num))\n\n        return True"
          }
        },
        {
          "id": 271,
          "title": "Encode and Decode Strings",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/encode-and-decode-strings/",
          "description": "Design an algorithm to serialize and deserialize a list of strings. The serialized string is then stored in a file or a database, and the deserialized string must reconstruct the original list.",
          "details": {
            "key_idea": null,
            "time_complexity": null,
            "space_complexity": null,
            "python_solution": "class Codec:\n    def encode(self, strs: List[str]) -> str:\n        encoded = \"\"\n        for s in strs:\n            encoded += str(len(s)) + \"#\" + s\n        return encoded\n\n    def decode(self, s: str) -> List[str]:\n        decoded = []\n        i = 0\n        while i < len(s):\n            delimiter_pos = s.find(\"#\", i)\n            size = int(s[i:delimiter_pos])\n            start_pos = delimiter_pos + 1\n            end_pos = start_pos + size\n            decoded.append(s[start_pos:end_pos])\n            i = end_pos\n        return decoded"
          }
        },
        {
          "id": 128,
          "title": "Longest Consecutive Sequence",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/longest-consecutive-sequence/",
          "description": "Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence. You must write an algorithm that runs in O(n) time.",
          "details": {
            "key_idea": "To find the longest consecutive subsequence, we first create a set of all the elements in the input array 'nums'. Then, for each element in the array, we check if it is the starting element of a consecutive subsequence. To do this, we check if the element before the current element (i.e., nums[i] - 1) exists in the set. If it doesn't, it means nums[i] is the starting element of a consecutive subsequence. From here, we keep incrementing the current element until the consecutive subsequence ends (i.e., the next element does not exist in the set). We keep track of the length of the consecutive subsequence and update the maximum length found so far.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def longestConsecutive(self, nums: List[int]) -> int:\n        num_set = set(nums)\n        max_length = 0\n\n        for num in num_set:\n            if num - 1 not in num_set:\n                current_num = num\n                current_length = 1\n\n                while current_num + 1 in num_set:\n                    current_num += 1\n                    current_length += 1\n\n                max_length = max(max_length, current_length)\n\n        return max_length"
          }
        }
      ]
    },
    {
      "topic_name": "Two Pointers",
      "problems": [
        {
          "id": 125,
          "title": "Valid Palindrome",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/valid-palindrome/",
          "description": "A phrase is a palindrome if it reads the same forward and backward, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters.",
          "details": {
            "key_idea": "To determine if a given string is a valid palindrome, we can use two pointers approach. We initialize two pointers, one at the beginning of the string (left) and the other at the end of the string (right). We then compare characters at these two pointers. If they are both alphanumeric characters and equal in value (ignoring case), we move both pointers towards the center of the string. If they are not equal, we know the string is not a palindrome. We continue this process until the two pointers meet or cross each other.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def isPalindrome(self, s: str) -> bool:\n        left, right = 0, len(s) - 1\n\n        while left < right:\n            while left < right and not s[left].isalnum():\n                left += 1\n            while left < right and not s[right].isalnum():\n                right -= 1\n\n            if s[left].lower() != s[right].lower():\n                return False\n\n            left += 1\n            right -= 1\n\n        return True"
          }
        },
        {
          "id": 167,
          "title": "Two Sum II - Input Array Is Sorted",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/",
          "description": "Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Return the indices of the two numbers added value (1-indexed). You may not use the same element twice. You must write an algorithm with only constant extra space complexity.",
          "details": {
            "key_idea": "The input array 'numbers' is already sorted in non-decreasing order. To find the two numbers that add up to the target, we can use a two-pointer approach. We initialize two pointers, one at the beginning of the array (left) and the other at the end of the array (right). We then check the sum of the elements at these two pointers. If the sum is equal to the target, we have found the pair. If the sum is less than the target, it means we need to increase the sum, so we move the left pointer one step to the right. If the sum is greater than the target, it means we need to decrease the sum, so we move the right pointer one step to the left. We continue this process until we find the pair or the two pointers meet or cross each other.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def twoSum(self, numbers: List[int], target: int) -> List[int]:\n        left, right = 0, len(numbers) - 1\n\n        while left < right:\n            current_sum = numbers[left] + numbers[right]\n\n            if current_sum == target:\n                return [left + 1, right + 1]\n            elif current_sum < target:\n                left += 1\n            else:\n                right -= 1\n\n        # No solution found\n        return [-1, -1]"
          }
        },
        {
          "id": 15,
          "title": "3Sum",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/3sum/",
          "description": "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.",
          "details": {
            "key_idea": "To find all unique triplets that sum to zero, we can use a three-pointer approach. First, we sort the input array 'nums' in non-decreasing order. Then, we iterate through the array with a fixed first element (i). For each fixed first element, we use two pointers (left and right) to find the other two elements that sum to the negation of the fixed first element. As the array is sorted, we can move these two pointers towards each other to efficiently find all possible triplets.",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def threeSum(self, nums: List[int]) -> List[List[int]]:\n        nums.sort()\n        result = []\n        n = len(nums)\n\n        for i in range(n - 2):\n            if i > 0 and nums[i] == nums[i - 1]:\n                continue\n\n            left, right = i + 1, n - 1\n\n            while left < right:\n                current_sum = nums[i] + nums[left] + nums[right]\n\n                if current_sum == 0:\n                    result.append([nums[i], nums[left], nums[right]])\n\n                    # Skip duplicates\n                    while left < right and nums[left] == nums[left + 1]:\n                        left += 1\n                    while left < right and nums[right] == nums[right - 1]:\n                        right -= 1\n\n                    left += 1\n                    right -= 1\n                elif current_sum < 0:\n                    left += 1\n                else:\n                    right -= 1\n\n        return result"
          }
        },
        {
          "id": 11,
          "title": "Container With Most Water",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/container-with-most-water/",
          "description": "You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]). Find two lines that together with the x-axis form a container, such that the container contains the most water.",
          "details": {
            "key_idea": "To find the maximum area of water that can be held between two vertical lines, we can use a two-pointer approach. We initialize two pointers, one at the beginning of the input array (left) and the other at the end of the array (right). The area between the two vertical lines is calculated as the minimum of the heights at the two pointers multiplied by the distance between them. We then update the maximum area found so far and move the pointer with the smaller height towards the other pointer. We continue this process until the two pointers meet or cross each other.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def maxArea(self, height: List[int]) -> int:\n        left, right = 0, len(height) - 1\n        max_area = 0\n\n        while left < right:\n            current_area = min(height[left], height[right]) * (right - left)\n            max_area = max(max_area, current_area)\n\n            if height[left] < height[right]:\n                left += 1\n            else:\n                right -= 1\n\n        return max_area"
          }
        },
        {
          "id": 42,
          "title": "Trapping Rain Water",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/trapping-rain-water/",
          "description": "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.",
          "details": {
            "key_idea": "To find the amount of trapped rainwater in the given elevation histogram represented by the input array 'height', we can use a two-pointer approach. We initialize two pointers, one at the beginning of the array (left) and the other at the end of the array (right). We also initialize two variables to keep track of the maximum left height and maximum right height seen so far. While the left pointer is less than the right pointer, we compare the height at the left and right pointers. If the height at the left pointer is less than or equal to the height at the right pointer, it means we can trap water between the left pointer and the maximum left height. Otherwise, we can trap water between the right pointer and the maximum right height. At each step, we update the trapped water amount and move the pointers and update the maximum heights accordingly.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def trap(self, height: List[int]) -> int:\n        left, right = 0, len(height) - 1\n        max_left, max_right = 0, 0\n        trapped_water = 0\n\n        while left < right:\n            if height[left] <= height[right]:\n                if height[left] >= max_left:\n                    max_left = height[left]\n                else:\n                    trapped_water += max_left - height[left]\n                left += 1\n            else:\n                if height[right] >= max_right:\n                    max_right = height[right]\n                else:\n                    trapped_water += max_right - height[right]\n                right -= 1\n\n        return trapped_water"
          }
        }
      ]
    },
    {
      "topic_name": "Sliding Window",
      "problems": [
        {
          "id": 121,
          "title": "Best Time to Buy & Sell Stock",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/best-time-to-buy-and-sell-stock/",
          "description": "You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.",
          "details": {
            "key_idea": "To find the maximum profit from a single buy and sell transaction in the input array 'prices', we can use a simple one-pass approach. We initialize two variables, 'min_price' to keep track of the minimum price seen so far, and 'max_profit' to store the maximum profit. We iterate through the 'prices' array, and for each price, we update the 'min_price' if we find a smaller price. We also calculate the potential profit if we sell at the current price and update 'max_profit' if the current profit is greater than the previous maximum profit.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def maxProfit(self, prices: List[int]) -> int:\n        if not prices:\n            return 0\n\n        min_price = float(\"inf\")\n        max_profit = 0\n\n        for price in prices:\n            min_price = min(min_price, price)\n            max_profit = max(max_profit, price - min_price)\n\n        return max_profit"
          }
        },
        {
          "id": 3,
          "title": "Longest Substring Without Repeating Characters",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/longest-substring-without-repeating-characters/",
          "description": "Given a string s, find the length of the longest substring without repeating characters.",
          "details": {
            "key_idea": "To find the length of the longest substring without repeating characters in the input string 's', we can use the sliding window approach. We use two pointers, 'left' and 'right', to represent the current window. As we move the 'right' pointer to the right, we expand the window and add characters to a set to keep track of unique characters in the window. If we encounter a repeating character, we move the 'left' pointer to the right to shrink the window until the repeating character is no longer in the window. At each step, we update the maximum length of the window (i.e., the length of the longest substring without repeating characters).",
            "time_complexity": "O(n)",
            "space_complexity": "O(k)",
            "python_solution": "class Solution:\n    def lengthOfLongestSubstring(self, s: str) -> int:\n        left, right = 0, 0\n        max_length = 0\n        unique_chars = set()\n\n        while right < len(s):\n            if s[right] not in unique_chars:\n                unique_chars.add(s[right])\n                max_length = max(max_length, right - left + 1)\n                right += 1\n            else:\n                unique_chars.remove(s[left])\n                left += 1\n\n        return max_length"
          }
        },
        {
          "id": 424,
          "title": "Longest Repeating Character Replacement",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/longest-repeating-character-replacement/",
          "description": "You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times. Return the length of the longest valid (string of same letter) substring after performing the above operations.",
          "details": {
            "key_idea": "To find the maximum length of a substring with at most k distinct characters in the input string 's', we can use the sliding window approach. We use two pointers, 'left' and 'right', to represent the current window. As we move the 'right' pointer to the right, we expand the window and add characters to a dictionary to keep track of their frequencies. If the number of distinct characters in the window exceeds k, we move the 'left' pointer to the right to shrink the window until the number of distinct characters is k again. At each step, we update the maximum length of the window.",
            "time_complexity": "O(n)",
            "space_complexity": "O(k)",
            "python_solution": "class Solution:\n    def characterReplacement(self, s: str, k: int) -> int:\n        left, right = 0, 0\n        max_length = 0\n        char_freq = {}\n        max_freq = 0\n\n        while right < len(s):\n            char_freq[s[right]] = char_freq.get(s[right], 0) + 1\n            max_freq = max(max_freq, char_freq[s[right]])\n\n            if (right - left + 1) - max_freq > k:\n                char_freq[s[left]] -= 1\n                left += 1\n\n            max_length = max(max_length, right - left + 1)\n            right += 1\n\n        return max_length"
          }
        },
        {
          "id": 567,
          "title": "Permutation in String",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/permutation-in-string/",
          "description": "Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise. In other words, return true if one of s1's permutations is the substring of s2.",
          "details": {
            "key_idea": "To check whether a string 's2' contains a permutation of another string 's1', we can use a sliding window approach. First, we create a frequency dictionary for characters in 's1'. Then, we initialize two pointers, 'left' and 'right', to represent the current window in 's2'. As we move the 'right' pointer to the right, we add the character to a temporary frequency dictionary and check if it becomes equal to the frequency dictionary of 's1'. If it does, it means we found a permutation of 's1' in 's2', and we return True. If the window size exceeds the length of 's1', we remove the character at the 'left' pointer from the temporary dictionary and move the 'left' pointer to the right to shrink the window. We continue this process until we find a permutation or reach the end of 's2'.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def checkInclusion(self, s1: str, s2: str) -> bool:\n        if len(s1) > len(s2):\n            return False\n\n        char_freq_s1 = {}\n        for char in s1:\n            char_freq_s1[char] = char_freq_s1.get(char, 0) + 1\n\n        left, right = 0, 0\n        char_freq_temp = {}\n\n        while right < len(s2):\n            char_freq_temp[s2[right]] = char_freq_temp.get(s2[right], 0) + 1\n\n            if right - left + 1 == len(s1):\n                if char_freq_temp == char_freq_s1:\n                    return True\n                char_freq_temp[s2[left]] -= 1\n                if char_freq_temp[s2[left]] == 0:\n                    del char_freq_temp[s2[left]]\n                left += 1\n\n            right += 1\n\n        return False"
          }
        },
        {
          "id": 76,
          "title": "Minimum Window Substring",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/minimum-window-substring/",
          "description": "Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string \"\". The testcases will be generated such that the answer is unique.",
          "details": {
            "key_idea": "To find the minimum window in the input string 's' that contains all characters from another string 't', we can use the sliding window approach. We first create a frequency dictionary for characters in 't'. Then, we initialize two pointers, 'left' and 'right', to represent the current window in 's'. As we move the 'right' pointer to the right, we add the character to a temporary frequency dictionary and check if it contains all characters from 't'. If it does, it means we found a valid window containing all characters from 't'. We update the minimum window length and move the 'left' pointer to the right to shrink the window. We continue this process until we find the minimum window or reach the end of 's'.",
            "time_complexity": "O(n)",
            "space_complexity": "O(k)",
            "python_solution": "class Solution:\n    def minWindow(self, s: str, t: str) -> str:\n        if not s or not t:\n            return \"\"\n\n        char_freq_t = {}\n        for char in t:\n            char_freq_t[char] = char_freq_t.get(char, 0) + 1\n\n        left, right = 0, 0\n        char_freq_temp = {}\n        required_chars = len(char_freq_t)\n        formed_chars = 0\n        min_length = float(\"inf\")\n        min_window = \"\"\n\n        while right < len(s):\n            char_freq_temp[s[right]] = char_freq_temp.get(s[right], 0) + 1\n\n            if (\n                s[right] in char_freq_t\n                and char_freq_temp[s[right]] == char_freq_t[s[right]]\n            ):\n                formed_chars += 1\n\n            while left <= right and formed_chars == required_chars:\n                if right - left + 1 < min_length:\n                    min_length = right - left + 1\n                    min_window = s[left : right + 1]\n\n                char_freq_temp[s[left]] -= 1\n                if (\n                    s[left] in char_freq_t\n                    and char_freq_temp[s[left]] < char_freq_t[s[left]]\n                ):\n                    formed_chars -= 1\n\n                left += 1\n\n            right += 1\n\n        return min_window"
          }
        },
        {
          "id": 239,
          "title": "Sliding Window Maximum",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/sliding-window-maximum/",
          "description": "Given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position and returns the maximum k numbers in the window. For example, If nums = [1,3,-1,-3,5,3,6,7], and k = 3, return the maximum sliding window as [3,3,5,5,6,7].",
          "details": {
            "key_idea": "To find the maximum sliding window of size 'k' in the input array 'nums', we can use a deque (double-ended queue). The deque will store the indices of elements in 'nums' such that the elements at these indices are in decreasing order. As we traverse the array 'nums', we add the current element to the deque, but before adding, we remove elements from the back of the deque that are smaller than the current element. This ensures that the front element of the deque will always be the maximum element in the window. At each step, we check if the front element's index is within the valid range of the current window. If it is not, we remove the front element from the deque. As we traverse the array, we can build the maximum sliding window using the elements stored in the deque.",
            "time_complexity": "O(n)",
            "space_complexity": "O(k)",
            "python_solution": "from collections import deque\n\n\nclass Solution:\n    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:\n        if not nums or k <= 0:\n            return []\n\n        result = []\n        window = deque()\n\n        for i, num in enumerate(nums):\n            while window and nums[window[-1]] < num:\n                window.pop()\n\n            window.append(i)\n\n            if i - window[0] >= k:\n                window.popleft()\n\n            if i >= k - 1:\n                result.append(nums[window[0]])\n\n        return result"
          }
        }
      ]
    },
    {
      "topic_name": "Stack",
      "problems": [
        {
          "id": 20,
          "title": "Valid Parentheses",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/valid-parentheses/",
          "description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
          "details": {
            "key_idea": "To determine if a given string of parentheses 's' is valid, we can use a stack data structure. We iterate through each character in 's', and if the character is an opening parenthesis ('(', '{', '['), we push it onto the stack. If the character is a closing parenthesis (')', '}', ']'), we check if the stack is empty or if the top element of the stack does not match the current closing parenthesis. If either of these conditions is met, we know the string is not valid. Otherwise, we pop the top element from the stack. At the end, if the stack is empty, the string is valid.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def isValid(self, s: str) -> bool:\n        stack = []\n        parentheses_map = {')': '(', '}': '{', ']': '['}\n\n        for char in s:\n            if char in parentheses_map.values():\n                stack.append(char)\n            elif char in parentheses_map:\n                if not stack or stack[-1] != parentheses_map[char]:\n                    return False\n                stack.pop()\n            else:\n                return False\n\n        return not stack"
          }
        },
        {
          "id": 155,
          "title": "Min Stack",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/min-stack/",
          "description": "Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.",
          "details": {
            "key_idea": "To implement a stack that supports finding the minimum element in constant time, we can use two stacks: one for storing the actual elements (stack) and another for keeping track of the minimum elements (min_stack). The min_stack will always have the current minimum element at the top. When pushing an element onto the stack, we compare it with the top element of the min_stack and push the smaller element onto the min_stack. When popping an element from the stack, we check if the element being popped is the same as the top element of the min_stack. If it is, we also pop the element from the min_stack. This way, the top element of the min_stack will always be the minimum element in the stack.",
            "time_complexity": "O(1)",
            "space_complexity": "O(n)",
            "python_solution": "class MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []\n\n    def push(self, val: int) -> None:\n        self.stack.append(val)\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n\n    def pop(self) -> None:\n        if self.stack:\n            if self.stack[-1] == self.min_stack[-1]:\n                self.min_stack.pop()\n            self.stack.pop()\n\n    def top(self) -> int:\n        if self.stack:\n            return self.stack[-1]\n\n    def getMin(self) -> int:\n        if self.min_stack:\n            return self.min_stack[-1]"
          }
        },
        {
          "id": 150,
          "title": "Evaluate Reverse Polish Notation",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/evaluate-reverse-polish-notation/",
          "description": "Evaluate the value of an arithmetic expression in Reverse Polish Notation. Valid operators are +, -, *, /. Each operand may be an integer or another expression. Note that division between two integers should truncate toward zero.",
          "details": {
            "key_idea": "To evaluate a given reverse Polish notation expression, we can use a stack data structure. We iterate through the tokens in the expression, and for each token, if it is a number, we push it onto the stack. If it is an operator ('+', '-', '*', '/'), we pop the top two elements from the stack, apply the operator to them, and push the result back onto the stack. At the end, the top element of the stack will be the final result of the expression.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def evalRPN(self, tokens: List[str]) -> int:\n        stack = []\n\n        for token in tokens:\n            if token.isdigit() or (token[0] == \"-\" and token[1:].isdigit()):\n                stack.append(int(token))\n            else:\n                num2 = stack.pop()\n                num1 = stack.pop()\n                if token == \"+\":\n                    stack.append(num1 + num2)\n                elif token == \"-\":\n                    stack.append(num1 - num2)\n                elif token == \"*\":\n                    stack.append(num1 * num2)\n                elif token == \"/\":\n                    stack.append(int(num1 / num2))\n\n        return stack[0]"
          }
        },
        {
          "id": 22,
          "title": "Generate Parentheses",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/generate-parentheses/",
          "description": "Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.",
          "details": {
            "key_idea": "To generate all valid combinations of parentheses, we can use backtracking. We start with an empty string and two counters, one for the open parentheses and one for the close parentheses. At each step, we have two choices: add an open parenthesis if the count of open parentheses is less than the total number of pairs, or add a close parenthesis if the count of close parentheses is less than the count of open parentheses. We continue this process recursively until we reach the desired length of the string. If the string becomes valid, we add it to the result.",
            "time_complexity": "O(4^n / sqrt(n))",
            "space_complexity": "O(4^n / sqrt(n))",
            "python_solution": "class Solution:\n    def generateParenthesis(self, n: int) -> List[str]:\n        def backtrack(s, open_count, close_count):\n            if len(s) == 2 * n:\n                result.append(s)\n                return\n\n            if open_count < n:\n                backtrack(s + \"(\", open_count + 1, close_count)\n            if close_count < open_count:\n                backtrack(s + \")\", open_count, close_count + 1)\n\n        result = []\n        backtrack(\"\", 0, 0)\n        return result"
          }
        },
        {
          "id": 739,
          "title": "Daily Temperatures",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/daily-temperatures/",
          "description": "Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.",
          "details": {
            "key_idea": "To find the daily temperatures that are warmer in the input array 'temperatures', we can use a stack. We iterate through the temperatures in reverse order, and for each temperature, we pop elements from the stack while they are smaller than the current temperature. This indicates that the current temperature is the first warmer temperature for the popped elements. We keep track of the indices of these warmer temperatures in the result array. Then, we push the current temperature's index onto the stack. At the end, the result array will contain the number of days until the next warmer temperature for each day.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:\n        stack = []\n        result = [0] * len(temperatures)\n\n        for i in range(len(temperatures) - 1, -1, -1):\n            while stack and temperatures[i] >= temperatures[stack[-1]]:\n                stack.pop()\n            if stack:\n                result[i] = stack[-1] - i\n            stack.append(i)\n\n        return result"
          }
        },
        {
          "id": 853,
          "title": "Car Fleet",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/car-fleet/",
          "description": "There are n cars going to the same destination along a one-lane road. The destination is at position target miles. You are given two integer arrays position and speed, both of length n. The ith car starts at position position[i] and has speed speed[i]. A car can never pass another car ahead of it, but it can catch up to it and drive bumper to bumper at the same speed. The faster car will slow down to match the slower car's speed. The distance between these two cars is ignored (they are assumed to be infinitely close). A car fleet is some non-empty set of cars driving at the same position and same speed. Note that a single car is also a car fleet. If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.",
          "details": {
            "key_idea": "To determine the number of car fleets that can reach the target destination in the input arrays 'target' and 'position', we can simulate the car movements and calculate the time it takes for each car to reach the target. We can then sort the cars based on their positions and iterate through them. For each car, we calculate its time to reach the target and compare it with the previous car. If the time for the current car is greater, it means the previous car cannot catch up to it, so we consider the current car as a new fleet. Otherwise, the previous car can catch up to the current car, so they form a fleet together.",
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:\n        cars = sorted(zip(position, speed), reverse=True)\n        fleets = 0\n        prev_time = -1.0\n\n        for pos, spd in cars:\n            time = (target - pos) / spd\n            if time > prev_time:\n                fleets += 1\n                prev_time = time\n\n        return fleets"
          }
        },
        {
          "id": 84,
          "title": "Largest Rectangle in Histogram",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/largest-rectangle-in-histogram/",
          "description": "Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.",
          "details": {
            "key_idea": "To find the largest rectangle area in the input histogram represented by the list 'heights', we can use a stack to keep track of increasing bar heights' indices. We iterate through the heights and push the current index onto the stack if the current height is greater than or equal to the height at the top of the stack. If the current height is smaller, it indicates that the previous bars cannot form a larger rectangle, so we pop indices from the stack and calculate the area for each popped bar. The width of the rectangle is determined by the difference between the current index and the index at the top of the stack. The height of the rectangle is the height at the popped index. We keep track of the maximum area seen so far.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def largestRectangleArea(self, heights: List[int]) -> int:\n        stack = []\n        max_area = 0\n\n        for i in range(len(heights)):\n            while stack and heights[i] < heights[stack[-1]]:\n                height = heights[stack.pop()]\n                width = i if not stack else i - stack[-1] - 1\n                max_area = max(max_area, height * width)\n            stack.append(i)\n\n        while stack:\n            height = heights[stack.pop()]\n            width = len(heights) if not stack else len(heights) - stack[-1] - 1\n            max_area = max(max_area, height * width)\n\n        return max_area"
          }
        }
      ]
    },
    {
      "topic_name": "Binary Search",
      "problems": [
        {
          "id": 704,
          "title": "Binary Search",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/binary-search/",
          "description": "Given an array of integers nums sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.",
          "details": {
            "key_idea": "Binary search is an efficient technique to search for a target element in a sorted array. In each step, we compare the middle element of the array with the target. If the middle element is equal to the target, we have found the element and return its index. If the middle element is greater than the target, we narrow down the search to the left half of the array. If the middle element is smaller, we narrow down the search to the right half of the array. We repeat this process until we find the target element or the search space is exhausted.",
            "time_complexity": "O(log n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def search(self, nums: List[int], target: int) -> int:\n        left, right = 0, len(nums) - 1\n\n        while left <= right:\n            mid = left + (right - left) // 2\n\n            if nums[mid] == target:\n                return mid\n            elif nums[mid] < target:\n                left = mid + 1\n            else:\n                right = mid - 1\n\n        return -1"
          }
        },
        {
          "id": 74,
          "title": "Search a 2D Matrix",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/search-a-2d-matrix/",
          "description": "You are given an m x n integer matrix matrix with the following two properties: Integers in each row are sorted in ascending from left to right. Integers in each column are sorted in ascending from top to bottom. Given an integer target, return true if target is in matrix or false otherwise.",
          "details": {
            "key_idea": "Since both the rows and columns in the input 2D matrix are sorted, we can treat the matrix as a one-dimensional sorted array and perform binary search to find the target element. We can map the 2D indices to the corresponding index in the 1D array and then apply binary search to locate the target element.",
            "time_complexity": "O(log(m*n))",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:\n        if not matrix or not matrix[0]:\n            return False\n\n        rows, cols = len(matrix), len(matrix[0])\n        left, right = 0, rows * cols - 1\n\n        while left <= right:\n            mid = left + (right - left) // 2\n            num = matrix[mid // cols][mid % cols]\n\n            if num == target:\n                return True\n            elif num < target:\n                left = mid + 1\n            else:\n                right = mid - 1\n\n        return False"
          }
        },
        {
          "id": 875,
          "title": "Koko Eating Bananas",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/koko-eating-bananas/",
          "description": "Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have set the exact hours Koko must eat them. If he eats k bananas on a given hour, he will eat k bananas from the current pile as the guards do not allow him to eat more than k bananas per hour. Each pile of bananas is completely eaten by Koko before moving on to the next pile. Koko likes to eat slower when he has guards around him. You are given a distinct integers hours, the maximum number of hours Koko has to eat all the bananas.",
          "details": {
            "key_idea": "The key idea is to perform binary search to find the minimum value of the integer 'k' such that Koko can eat all the bananas within 'hours' hours. We can define a binary search space for 'k' and perform binary search to find the smallest 'k' that satisfies the given condition.",
            "time_complexity": "O(n * log(max_pile))",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def minEatingSpeed(self, piles: List[int], h: int) -> int:\n        left, right = 1, max(piles)\n\n        while left < right:\n            mid = left + (right - left) // 2\n            hours = sum((pile + mid - 1) // mid for pile in piles)\n\n            if hours > h:\n                left = mid + 1\n            else:\n                right = mid\n\n        return left"
          }
        },
        {
          "id": 33,
          "title": "Search in Rotated Sorted Array",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/search-in-rotated-sorted-array/",
          "description": "There is an integer array nums sorted in ascending order (with distinct values). Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]. Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.",
          "details": {
            "key_idea": "The key idea is to perform binary search to find the target element in the rotated sorted array. We compare the middle element with the target and the endpoints of the subarray to determine which part of the array is sorted. Depending on the comparison, we narrow down the search to the sorted part of the array.",
            "time_complexity": "O(log n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def search(self, nums: List[int], target: int) -> int:\n        left, right = 0, len(nums) - 1\n\n        while left <= right:\n            mid = left + (right - left) // 2\n\n            if nums[mid] == target:\n                return mid\n\n            if nums[left] <= nums[mid]:\n                if nums[left] <= target < nums[mid]:\n                    right = mid - 1\n                else:\n                    left = mid + 1\n            else:\n                if nums[mid] < target <= nums[right]:\n                    left = mid + 1\n                else:\n                    right = mid - 1\n\n        return -1"
          }
        },
        {
          "id": 153,
          "title": "Find Minimum in Rotated Sorted Array",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/",
          "description": "Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]). Find the minimum element. You may assume no duplicate exists in the array.",
          "details": {
            "key_idea": "The key idea is to perform binary search to find the minimum element in the rotated sorted array. We compare the middle element with its neighbors to determine if it is the minimum element. Depending on the comparison, we narrow down the search to the unsorted part of the array.",
            "time_complexity": "O(log n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def findMin(self, nums: List[int]) -> int:\n        left, right = 0, len(nums) - 1\n\n        while left < right:\n            mid = left + (right - left) // 2\n\n            if nums[mid] > nums[right]:\n                left = mid + 1\n            else:\n                right = mid\n\n        return nums[left]"
          }
        },
        {
          "id": 981,
          "title": "Time Based Key-Value Store",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/time-based-key-value-store/",
          "description": "Create a time-based key-value store class that stores strings key value pairs, along with timestamp. There is a function store(key, value, timestamp), which stores the key value pair, and a function retrieve(key, timestamp), which returns the value associated with key such that its timestamp is less than or equal to the given timestamp. If there are multiple such values, it returns the value associated with the largest timestamp. If there are no such values, it returns the empty string.",
          "details": {
            "key_idea": "To implement a time-based key-value store, we can use a dictionary to store the values associated with each key. For each key, we store a list of tuples representing the timestamp and the corresponding value. When querying a key at a specific timestamp, we perform binary search on the list of timestamps associated with that key to find the largest timestamp less than or equal to the given timestamp.",
            "time_complexity": "set: O(1), get: O(log n)",
            "space_complexity": "O(n)",
            "python_solution": "from collections import defaultdict\n\n\nclass TimeMap:\n    def __init__(self):\n        self.data = defaultdict(list)\n\n    def set(self, key: str, value: str, timestamp: int) -> None:\n        self.data[key].append((timestamp, value))\n\n    def get(self, key: str, timestamp: int) -> str:\n        values = self.data[key]\n        left, right = 0, len(values) - 1\n\n        while left <= right:\n            mid = left + (right - left) // 2\n            if values[mid][0] == timestamp:\n                return values[mid][1]\n            elif values[mid][0] < timestamp:\n                left = mid + 1\n            else:\n                right = mid - 1\n\n        if right >= 0:\n            return values[right][1]\n        return \"\""
          }
        },
        {
          "id": 4,
          "title": "Median of Two Sorted Arrays",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/median-of-two-sorted-arrays/",
          "description": "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).",
          "details": {
            "key_idea": "To find the median of two sorted arrays 'nums1' and 'nums2', we can perform a binary search on the smaller array. We partition both arrays into two parts such that the left half contains smaller elements and the right half contains larger elements. The median will be the average of the maximum element in the left half and the minimum element in the right half. We adjust the partition indices based on binary search, aiming to keep the same number of elements in both halves.",
            "time_complexity": "O(log(min(m, n)))",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:\n        if len(nums1) > len(nums2]):\n            nums1, nums2 = nums2, nums1\n\n        m, n = len(nums1), len(nums2)\n        low, high = 0, m\n\n        while low <= high:\n            partition1 = (low + high) // 2\n            partition2 = (m + n + 1) // 2 - partition1\n\n            maxLeft1 = float(\"-inf\") if partition1 == 0 else nums1[partition1 - 1]\n            minRight1 = float(\"inf\") if partition1 == m else nums1[partition1]\n\n            maxLeft2 = float(\"-inf\") if partition2 == 0 else nums2[partition2 - 1]\n            minRight2 = float(\"inf\") if partition2 == n else nums2[partition2]\n\n            if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:\n                if (m + n) % 2 == 0:\n                    return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2\n                else:\n                    return max(maxLeft1, maxLeft2)\n            elif maxLeft1 > minRight2:\n                high = partition1 - 1\n            else:\n                low = partition1 + 1\n\n        raise ValueError(\"Input arrays are not sorted.\")"
          }
        }
      ]
    },
    {
      "topic_name": "Linked List",
      "problems": [
        {
          "id": 206,
          "title": "Reverse Linked List",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/reverse-linked-list/",
          "description": "Given the head of a singly linked list, reverse the list, and return the reversed list.",
          "details": {
            "key_idea": "To reverse a singly linked list, we need to reverse the direction of the pointers while traversing the list. We maintain three pointers: 'prev' (to keep track of the previous node), 'current' (to keep track of the current node), and 'next_node' (to keep track of the next node in the original list). In each iteration, we update the 'current.next' pointer to point to the 'prev' node, and then move 'prev' and 'current' pointers one step forward. We repeat this process until we reach the end of the original list, and the 'prev' pointer will be pointing to the new head of the reversed list.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\n\n\nclass Solution:\n    def reverseList(self, head: ListNode) -> ListNode:\n        prev = None\n        current = head\n\n        while current:\n            next_node = current.next\n            current.next = prev\n            prev = current\n            current = next_node\n\n        return prev"
          }
        },
        {
          "id": 21,
          "title": "Merge Two Sorted Lists",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/merge-two-sorted-lists/",
          "description": "You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists. Return the head of the merged linked list.",
          "details": {
            "key_idea": "To merge two sorted linked lists 'l1' and 'l2', we can create a new linked list 'dummy' to hold the merged result. We maintain two pointers, 'current' and 'prev', to traverse through the two input lists. At each step, we compare the values at the 'current' pointers of 'l1' and 'l2', and add the smaller value to the 'dummy' list. We then move the 'current' pointer of the list with the smaller value one step forward. After iterating through both lists, if any list still has remaining elements, we append them to the 'dummy' list.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\n\n\nclass Solution:\n    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:\n        dummy = ListNode()\n        current = dummy\n\n        while l1 and l2:\n            if l1.val < l2.val:\n                current.next = l1\n                l1 = l1.next\n            else:\n                current.next = l2\n                l2 = l2.next\n            current = current.next\n\n        if l1:\n            current.next = l1\n        elif l2:\n            current.next = l2\n\n        return dummy.next"
          }
        },
        {
          "id": 143,
          "title": "Reorder List",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/reorder-list/",
          "description": "You are given the head of a singly linked list. The list of nodes is labeled from 1 to n. We want to reorder the list such that for the given list: 1->2->3->4->5, we want to reorder it to 1->5->2->4->3.",
          "details": {
            "key_idea": "To reorder a singly linked list, we can break the list into two halves, reverse the second half, and then merge the two halves alternatively. First, we find the middle of the list using the slow and fast pointer technique. We reverse the second half of the list in place. Finally, we merge the two halves by alternating nodes from each half.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\n\n\nclass Solution:\n    def reorderList(self, head: ListNode) -> None:\n        if not head or not head.next or not head.next.next:\n            return\n\n        # Find the middle of the list\n        slow, fast = head, head\n        while fast.next and fast.next.next:\n            slow = slow.next\n            fast = fast.next.next\n\n        # Reverse the second half of the list\n        prev, current = None, slow.next\n        slow.next = None\n        while current:\n            next_node = current.next\n            current.next = prev\n            prev = current\n            current = next_node\n\n        # Merge the two halves alternately\n        p1, p2 = head, prev\n        while p2:\n            next_p1, next_p2 = p1.next, p2.next\n            p1.next = p2\n            p2.next = next_p1\n            p1, p2 = next_p1, next_p2"
          }
        },
        {
          "id": 19,
          "title": "Remove Nth Node From End of List",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/remove-nth-node-from-end-of-list/",
          "description": "Given the head of a linked list, remove the nth node from the end of the list and return its head.",
          "details": {
            "key_idea": "To remove the nth node from the end of a singly linked list, we can use the two-pointer approach. We maintain two pointers, 'fast' and 'slow', where 'fast' moves n nodes ahead of 'slow'. Then we move both pointers simultaneously until 'fast' reaches the end of the list. At this point, 'slow' will be pointing to the node just before the node to be removed. We update the 'slow.next' pointer to skip the node to be removed.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\n\n\nclass Solution:\n    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:\n        dummy = ListNode(0)\n        dummy.next = head\n        fast = slow = dummy\n\n        # Move 'fast' n nodes ahead\n        for _ in range(n):\n            fast = fast.next\n\n        # Move both pointers until 'fast' reaches the end\n        while fast.next:\n            fast = fast.next\n            slow = slow.next\n\n        # Remove the nth node from the end\n        slow.next = slow.next.next\n\n        return dummy.next"
          }
        },
        {
          "id": 138,
          "title": "Copy List with Random Pointer",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/copy-list-with-random-pointer/",
          "description": "A linked list of size n, where each node has a next pointer and a random pointer, return a deep copy of the list. Each node's value is between -10000 and 10000, and the number of nodes is n. The value of each node is not necessarily unique, and the random pointer could point to any node in the list or null.",
          "details": {
            "key_idea": "To create a deep copy of a linked list with random pointers, we can follow a three-step approach. First, we duplicate each node in the original list and insert the duplicates right after their corresponding original nodes. Second, we update the random pointers of the duplicate nodes to point to the correct nodes. Finally, we split the combined list into two separate lists: the original list and the duplicated list.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "# Definition for a Node.\n# class Node:\n#     def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):\n#         self.val = int(x)\n#         self.next = next\n#         self.random = random\n\n\nclass Solution:\n    def copyRandomList(self, head: \"Node\") -> \"Node\":\n        if not head:\n            return None\n\n        # Step 1: Duplicate nodes and insert them in between the original nodes\n        current = head\n        while current:\n            duplicate = Node(current.val)\n            duplicate.next = current.next\n            current.next = duplicate\n            current = duplicate.next\n\n        # Step 2: Update random pointers for the duplicate nodes\n        current = head\n        while current:\n            if current.random:\n                current.next.random = current.random.next\n            current = current.next.next\n\n        # Step 3: Split the combined list into two separate lists\n        original = head\n        duplicate_head = head.next\n        current = duplicate_head\n        while original:\n            original.next = original.next.next\n            if current.next:\n                current.next = current.next.next\n            original = original.next\n            current = current.next\n\n        return duplicate_head"
          }
        },
        {
          "id": 2,
          "title": "Add Two Numbers",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/add-two-numbers/",
          "description": "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.",
          "details": {
            "key_idea": "To add two numbers represented by linked lists, we can simulate the addition digit by digit while considering carry. We maintain a dummy node to build the resulting linked list. We iterate through the input lists, summing the corresponding digits along with any carry from the previous digit. We update the carry and create a new node with the sum digit. After processing both lists, if there is a carry remaining, we add a new node with the carry.",
            "time_complexity": "O(max(m, n))",
            "space_complexity": "O(max(m, n))",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\n\n\nclass Solution:\n    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:\n        dummy = ListNode()\n        current = dummy\n        carry = 0\n\n        while l1 or l2:\n            val1 = l1.val if l1 else 0\n            val2 = l2.val if l2 else 0\n            total = val1 + val2 + carry\n\n            carry = total // 10\n            digit = total % 10\n\n            current.next = ListNode(digit)\n            current = current.next\n\n            if l1:\n                l1 = l1.next\n            if l2:\n                l2 = l2.next\n\n        if carry:\n            current.next = ListNode(carry)\n\n        return dummy.next"
          }
        },
        {
          "id": 141,
          "title": "Linked List Cycle",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/linked-list-cycle/",
          "description": "Given head, the head of a linked list, determine if the linked list has a cycle in it.",
          "details": {
            "key_idea": "To detect a cycle in a linked list, we can use the Floyd's Tortoise and Hare algorithm. We maintain two pointers, 'slow' and 'fast', where 'slow' moves one step at a time and 'fast' moves two steps at a time. If there is a cycle in the linked list, the two pointers will eventually meet at some point. If there is no cycle, the 'fast' pointer will reach the end of the list.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, x):\n#         self.val = x\n#         self.next = None\n\n\nclass Solution:\n    def hasCycle(self, head: ListNode) -> bool:\n        if not head or not head.next:\n            return False\n\n        slow = head\n        fast = head.next\n\n        while slow != fast:\n            if not fast or not fast.next:\n                return False\n            slow = slow.next\n            fast = fast.next.next\n\n        return True"
          }
        },
        {
          "id": 287,
          "title": "Find the Duplicate Number",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/find-the-duplicate-number/",
          "description": "Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive. There is only one number in the range that appears twice or more, return its duplicate number. You must solve the problem without modifying the array nums and uses only constant extra space.",
          "details": {
            "key_idea": "To find the duplicate number in an array, we can treat the array as a linked list where each value points to the next value in the array. This problem is then reduced to finding the cycle in the linked list. We use the Floyd's Tortoise and Hare algorithm to detect the cycle. Once the cycle is detected, we find the entrance of the cycle, which represents the duplicate number.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def findDuplicate(self, nums: List[int]) -> int:\n        slow = nums[0]\n        fast = nums[0]\n\n        # Move slow and fast pointers\n        while True:\n            slow = nums[slow]\n            fast = nums[nums[fast]]\n            if slow == fast:\n                break\n\n        # Find the entrance of the cycle\n        slow = nums[0]\n        while slow != fast:\n            slow = nums[slow]\n            fast = nums[fast]\n\n        return slow"
          }
        },
        {
          "id": 146,
          "title": "LRU Cache",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/lru-cache/",
          "description": "Design a data structure that follows the constraints of a Least Recently Used (LRU) cache. Implement the LRUCache class: LRUCache(int capacity) Initialize the LRU cache with positive size capacity. int get(int key) Return the value of the key if the key exists, otherwise return -1. void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.",
          "details": {
            "key_idea": "To implement an LRU (Least Recently Used) cache, we can use a combination of a dictionary (to store key-value pairs) and a doubly linked list (to maintain the order of usage). The dictionary allows for quick access to values, and the doubly linked list helps in efficient removal and addition of elements. When a key is accessed or a new key is added, we update its position in the linked list. When the cache is full, we remove the least recently used item from the tail of the linked list.",
            "time_complexity": "get: O(1), put: O(1)",
            "space_complexity": "O(capacity)",
            "python_solution": "from collections import OrderedDict\n\nclass LRUCache:\n\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n\n    def get(self, key: int) -> int:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n            return self.cache[key]\n        else:\n            return -1\n\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        elif self.capacity <= 0:\n            _ = self.cache.popitem(False)\n        else:\n            self.capacity = max(0, self.capacity - 1)\n        self.cache[key] = value"
          }
        },
        {
          "id": 23,
          "title": "Merge k Sorted Lists",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/merge-k-sorted-lists/",
          "description": "You are given an array of k linked-list lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
          "details": {
            "key_idea": "To merge k sorted linked lists, we can use a min-heap (priority queue) to keep track of the smallest element from each list. We initially add the first element from each list to the heap. Then, in each iteration, we pop the smallest element from the heap and add it to the merged result. If the popped element has a next element in its original list, we add that next element to the heap. We continue this process until the heap is empty.",
            "time_complexity": "O(N log k)",
            "space_complexity": "O(k)",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\n\nimport heapq\n\n\nclass Solution:\n    def mergeKLists(self, lists: List[ListNode]) -> ListNode:\n        min_heap = []\n        for i, l in enumerate(lists):\n            if l:\n                heapq.heappush(min_heap, (l.val, i))\n\n        dummy = ListNode()\n        current = dummy\n\n        while min_heap:\n            val, idx = heapq.heappop(min_heap)\n            current.next = ListNode(val)\n            current = current.next\n            if lists[idx].next:\n                heapq.heappush(min_heap, (lists[idx].next.val, idx))\n                lists[idx] = lists[idx].next\n\n        return dummy.next"
          }
        },
        {
          "id": 25,
          "title": "Reverse Nodes in k-Group",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/reverse-nodes-in-k-group/",
          "description": "Given the head of a linked list, reverse the nodes of the list k at a time and return the modified list. k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.",
          "details": {
            "key_idea": "To reverse nodes in k-group, we can use a recursive approach. We traverse the linked list in groups of k nodes, reversing each group. For each group, we maintain pointers to the group's first node ('start') and the group's last node ('end'). We reverse the group in-place and connect the previous group's 'end' to the reversed group's 'start'. We then recursively reverse the remaining part of the linked list.",
            "time_complexity": "O(n)",
            "space_complexity": "O(k)",
            "python_solution": "# Definition for singly-linked list.\n# class ListNode:\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\n\n\nclass Solution:\n    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:\n        if not head or k == 1:\n            return head\n\n        # Count the number of nodes in the list\n        count = 0\n        current = head\n        while current:\n            count += 1\n            current = current.next\n\n        if count < k:\n            return head\n\n        # Reverse the first k nodes\n        prev, current = None, head\n        for _ in range(k):\n            next_node = current.next\n            current.next = prev\n            prev = current\n            current = next_node\n\n        # Recursively reverse the remaining part of the list\n        head.next = self.reverseKGroup(current, k)\n\n        return prev"
          }
        }
      ]
    },
    {
      "topic_name": "Trees",
      "problems": [
        {
          "id": 226,
          "title": "Invert Binary Tree",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/invert-binary-tree/",
          "description": "Given the root of a binary tree, invert the tree, and return its root.",
          "details": {
            "key_idea": "To invert a binary tree, we can use a recursive approach. For each node, we swap its left and right subtrees, and then recursively invert the left and right subtrees.",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def invertTree(self, root: TreeNode) -> TreeNode:\n        if not root:\n            return None\n\n        # Swap left and right subtrees\n        root.left, root.right = root.right, root.left\n\n        # Recursively invert left and right subtrees\n        self.invertTree(root.left)\n        self.invertTree(root.right)\n\n        return root"
          }
        },
        {
          "id": 104,
          "title": "Maximum Depth of Binary Tree",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/maximum-depth-of-binary-tree/",
          "description": "Given the root of a binary tree, return its maximum depth.",
          "details": {
            "key_idea": "To find the maximum depth of a binary tree, we can use a recursive approach. For each node, the maximum depth is the maximum of the depths of its left and right subtrees, plus one. We start from the root and recursively calculate the maximum depth for each subtree.",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def maxDepth(self, root: TreeNode) -> int:\n        if not root:\n            return 0\n\n        left_depth = self.maxDepth(root.left)\n        right_depth = self.maxDepth(root.right)\n\n        return max(left_depth, right_depth) + 1"
          }
        },
        {
          "id": 543,
          "title": "Diameter of Binary Tree",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/diameter-of-binary-tree/",
          "description": "Given the root of a binary tree, return the length of the longest path between any two nodes in the tree. This path may or may not pass through the root.",
          "details": {
            "key_idea": "To find the diameter of a binary tree (the length of the longest path between any two nodes), we can use a recursive approach. For each node, the longest path passes either through the node or doesn't. The diameter is the maximum of three values: the diameter of the left subtree, the diameter of the right subtree, and the sum of the heights of the left and right subtrees (if the path passes through the node).",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def diameterOfBinaryTree(self, root: TreeNode) -> int:\n        def height(node):\n            if not node:\n                return 0\n            left_height = height(node.left)\n            right_height = height(node.right)\n            self.diameter = max(self.diameter, left_height + right_height)\n            return max(left_height, right_height) + 1\n\n        self.diameter = 0\n        height(root)\n        return self.diameter"
          }
        },
        {
          "id": 110,
          "title": "Balanced Binary Tree",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/balanced-binary-tree/",
          "description": "Given a binary tree root, determine if it is a height-balanced binary tree.",
          "details": {
            "key_idea": "To check if a binary tree is balanced, we can use a recursive approach. For each node, we calculate the height of its left and right subtrees. If the difference in heights is greater than 1, the tree is not balanced. We continue this process for all nodes, recursively checking each subtree.",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def isBalanced(self, root: TreeNode) -> bool:\n        def height(node):\n            if not node:\n                return 0\n            left_height = height(node.left)\n            right_height = height(node.right)\n            if abs(left_height - right_height) > 1:\n                return float(\"inf\")  # Indicate imbalance\n            return max(left_height, right_height) + 1\n\n        return height(root) != float(\"inf\")"
          }
        },
        {
          "id": 100,
          "title": "Same Tree",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/same-tree/",
          "description": "Given the roots of two binary trees p and q, write a function to check if they are the same or not.",
          "details": {
            "key_idea": "To determine if two binary trees are the same, we can use a recursive approach. For each pair of corresponding nodes, we compare their values and recursively check the left and right subtrees. If the values are equal and the left and right subtrees are also equal, then the trees are the same.",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:\n        if not p and not q:\n            return True\n        if not p or not q or p.val != q.val:\n            return False\n        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)"
          }
        },
        {
          "id": 572,
          "title": "Subtree Of Another Tree",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/subtree-of-another-tree/",
          "description": "Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values as subRoot or false otherwise.",
          "details": {
            "key_idea": "To check if one binary tree is a subtree of another, we can use a recursive approach. For each node in the main tree, we check if the current subtree rooted at that node is equal to the given subtree. If not, we recursively check the left and right subtrees.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:\n        if not s:\n            return False\n        if self.isSameTree(s, t):\n            return True\n        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)\n\n    def isSameTree(self, p, q):\n        if not p and not q:\n            return True\n        if not p or not q or p.val != q.val:\n            return False\n        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)"
          }
        },
        {
          "id": 235,
          "title": "Lowest Common Ancestor Of A Binary Search Tree",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/",
          "description": "Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.",
          "details": {
            "key_idea": "To find the lowest common ancestor (LCA) of two nodes in a binary search tree (BST), we can use a recursive approach. We compare the values of the two nodes with the current node's value. If both nodes are in the left subtree, we move to the left child. If both nodes are in the right subtree, we move to the right child. If one node is in the left subtree and the other is in the right subtree, we've found the LCA.",
            "time_complexity": "O(h)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, x):\n#         self.val = x\n#         self.left = None\n#         self.right = None\n\n\nclass Solution:\n    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:\n        if root.val > p.val and root.val > q.val:\n            return self.lowestCommonAncestor(root.left, p, q)\n        elif root.val < p.val and root.val < q.val:\n            return self.lowestCommonAncestor(root.right, p, q)\n        else:\n            return root"
          }
        },
        {
          "id": 102,
          "title": "Binary Tree Level Order Traversal",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/binary-tree-level-order-traversal/",
          "description": "Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).",
          "details": {
            "key_idea": "To perform level order traversal of a binary tree, we can use a breadth-first search (BFS) approach. We start with the root node, and in each iteration, we process all nodes at the current level before moving to the next level. We use a queue to keep track of nodes at each level.",
            "time_complexity": "O(n)",
            "space_complexity": "O(w)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def levelOrder(self, root: TreeNode) -> List[List[int]]:\n        if not root:\n            return []\n\n        result = []\n        queue = [root]\n\n        while queue:\n            level = []\n            next_level = []\n\n            for node in queue:\n                level.append(node.val)\n                if node.left:\n                    next_level.append(node.left)\n                if node.right:\n                    next_level.append(node.right)\n\n            result.append(level)\n            queue = next_level\n\n        return result"
          }
        },
        {
          "id": 199,
          "title": "Binary Tree Right Side View",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/binary-tree-right-side-view/",
          "description": "Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.",
          "details": {
            "key_idea": "To obtain the right side view of a binary tree, we can perform a level order traversal using a breadth-first search (BFS) approach. For each level, we add the last node's value to the result list. This way, we capture the rightmost nodes at each level.",
            "time_complexity": "O(n)",
            "space_complexity": "O(w)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def rightSideView(self, root: TreeNode) -> List[int]:\n        if not root:\n            return []\n\n        result = []\n        queue = [root]\n\n        while queue:\n            level_size = len(queue)\n\n            for i in range(level_size):\n                node = queue.pop(0)\n                if i == level_size - 1:\n                    result.append(node.val)\n                if node.left:\n                    queue.append(node.left)\n                if node.right:\n                    queue.append(node.right)\n\n        return result"
          }
        },
        {
          "id": 1448,
          "title": "Count Good Nodes in Binary Tree",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/count-good-nodes-in-binary-tree/",
          "description": "Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.",
          "details": {
            "key_idea": "To count the number of good nodes in a binary tree, we can use a recursive depth-first search (DFS) approach. For each node, we keep track of the maximum value encountered on the path from the root to the current node. If the value of the current node is greater than or equal to the maximum value on the path, it is a good node. We increment the count and continue the DFS for the left and right subtrees.",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def goodNodes(self, root: TreeNode) -> int:\n        def dfs(node, max_val):\n            if not node:\n                return 0\n\n            if node.val >= max_val:\n                max_val = node.val\n                count = 1\n            else:\n                count = 0\n\n            count += dfs(node.left, max_val)\n            count += dfs(node.right, max_val)\n\n            return count\n\n        return dfs(root, float(\"-inf\"))"
          }
        },
        {
          "id": 98,
          "title": "Validate Binary Search Tree",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/validate-binary-search-tree/",
          "description": "Given the root of a binary tree, determine if it is a valid binary search tree (BST). A valid BST is defined as follows: The left subtree of a node contains only nodes with keys less than the node's key. The right subtree of a node contains only nodes with keys greater than the node's key. Both the left and right subtrees must also be binary search trees.",
          "details": {
            "key_idea": "To validate if a binary tree is a valid binary search tree (BST), we can perform an in-order traversal and check if the values are in ascending order. During the in-order traversal, we keep track of the previously visited node's value and compare it with the current node's value.",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def isValidBST(self, root: TreeNode) -> bool:\n        def inorder_traversal(node, prev):\n            if not node:\n                return True\n\n            if not inorder_traversal(node.left, prev):\n                return False\n\n            if prev[0] is not None and node.val <= prev[0]:\n                return False\n            prev[0] = node.val\n\n            return inorder_traversal(node.right, prev)\n\n        prev = [None]\n        return inorder_traversal(root, prev)"
          }
        },
        {
          "id": 230,
          "title": "Kth Smallest Element in a BST",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/kth-smallest-element-in-a-bst/",
          "description": "Given the root of a binary search tree and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.",
          "details": {
            "key_idea": "To find the kth smallest element in a binary search tree (BST), we can perform an in-order traversal and keep track of the count of visited nodes. When the count reaches k, we've found the kth smallest element.",
            "time_complexity": "O(h + k)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def kthSmallest(self, root: TreeNode, k: int) -> int:\n        def inorder_traversal(node):\n            if not node:\n                return []\n\n            left = inorder_traversal(node.left)\n            right = inorder_traversal(node.right)\n\n            return left + [node.val] + right\n\n        inorder_values = inorder_traversal(root)\n        return inorder_values[k - 1]"
          }
        },
        {
          "id": 105,
          "title": "Construct Binary Tree from Preorder and Inorder Traversal",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/",
          "description": "Given two integer arrays preorder and inorder where preorder is the preorder traversal of the tree and inorder is the inorder traversal of the tree, construct the binary tree, and return its root.",
          "details": {
            "key_idea": "To construct a binary tree from its preorder and inorder traversals, we can use a recursive approach. The first element in the preorder list is the root of the current subtree. We locate its position in the inorder list to determine the left and right subtrees. We recursively construct the left and right subtrees for each subtree.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:\n        if not preorder or not inorder:\n            return None\n\n        root_val = preorder.pop(0)\n        root = TreeNode(root_val)\n        root_index = inorder.index(root_val)\n\n        root.left = self.buildTree(preorder, inorder[:root_index])\n        root.right = self.buildTree(preorder, inorder[root_index + 1 :])\n\n        return root"
          }
        },
        {
          "id": 124,
          "title": "Binary Tree Maximum Path Sum",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/binary-tree-maximum-path-sum/",
          "description": "A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can have as many as two neighbors but no other number of neighbors. The length of a path is the sum of the node's values in the path. Given the root of a binary tree, return the maximum path sum of any non-empty path ending at a node in the tree.",
          "details": {
            "key_idea": "To find the maximum path sum in a binary tree, we can use a recursive approach. For each node, we calculate the maximum path sum that includes that node. This can be either the node's value itself or the value plus the maximum path sum from its left and right subtrees. We update the global maximum as we traverse the tree.",
            "time_complexity": "O(n)",
            "space_complexity": "O(h)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode:\n#     def __init__(self, val=0, left=None, right=None):\n#         self.val = val\n#         self.left = left\n#         self.right = right\n\n\nclass Solution:\n    def maxPathSum(self, root: TreeNode) -> int:\n        def maxPathSumHelper(node):\n            if not node:\n                return 0\n\n            left_sum = max(0, maxPathSumHelper(node.left))\n            right_sum = max(0, maxPathSumHelper(node.right))\n\n            self.max_sum = max(self.max_sum, left_sum + right_sum + node.val)\n\n            return max(left_sum, right_sum) + node.val\n\n        self.max_sum = float(\"-inf\")\n        maxPathSumHelper(root)\n\n        return self.max_sum"
          }
        },
        {
          "id": 297,
          "title": "Serialize and Deserialize Binary Tree",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/serialize-and-deserialize-binary-tree/",
          "description": "Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment. Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that the serialized string is a valid representation of a binary tree.",
          "details": {
            "key_idea": "To serialize a binary tree, we can perform a preorder traversal and serialize the nodes into a string. When deserializing, we split the string into a list of values and reconstruct the binary tree using a recursive approach.",
            "time_complexity": "Serialization: O(n), Deserialization: O(n)",
            "space_complexity": "Serialization: O(n), Deserialization: O(n)",
            "python_solution": "# Definition for a binary tree node.\n# class TreeNode(object):\n#     def __init__(self, x):\n#         self.val = x\n#         self.left = None\n#         self.right = None\n\n\nclass Codec:\n\n    def serialize(self, root):\n        \"\"\"Encodes a tree to a single string.\n\n        :type root: TreeNode\n        :rtype: str\n        \"\"\"\n\n        def preorder(node):\n            if not node:\n                return \"None,\"\n            return str(node.val) + \",\" + preorder(node.left) + preorder(node.right)\n\n        return preorder(root)\n\n    def deserialize(self, data):\n        \"\"\"Decodes your encoded data to tree.\n\n        :type data: str\n        :rtype: TreeNode\n        \"\"\"\n\n        def build_tree(values):\n            if values[0] == \"None\":\n                values.pop(0)\n                return None\n\n            root = TreeNode(int(values.pop(0)))\n            root.left = build_tree(values)\n            root.right = build_tree(values)\n\n            return root\n\n        values = data.split(\",\")\n        return build_tree(values[:-1])\n\n\n# Your Codec object will be instantiated and called as such:\n# ser = Codec()\n# deser = Codec()\n# ans = deser.deserialize(ser.serialize(root))"
          }
        }
      ]
    },
    {
      "topic_name": "Tries",
      "problems": [
        {
          "id": 208,
          "title": "Implement Trie (Prefix Tree)",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/implement-trie-prefix-tree/",
          "description": "Implement a trie with insert, search, and startsWith methods.",
          "details": {
            "key_idea": "To implement a Trie (prefix tree), we create a TrieNode class that represents each node in the trie. Each node contains a dictionary that maps characters to child nodes. We start with an empty root node and add words by traversing the characters and creating nodes as needed.",
            "time_complexity": "Insertion: O(m), Search: O(m), StartsWith: O(m)",
            "space_complexity": "O(n * m)",
            "python_solution": "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n\n    def insert(self, word: str) -> None:\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.is_end = True\n\n    def search(self, word: str) -> bool:\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return node.is_end\n\n    def startsWith(self, prefix: str) -> bool:\n        node = self.root\n        for char in prefix:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return True\n\n\n# Your Trie object will be instantiated and called as such:\n# obj = Trie()\n# obj.insert(word)\n# param_2 = obj.search(word)\n# param_3 = obj.startsWith(prefix)"
          }
        },
        {
          "id": 211,
          "title": "Design Add and Search Words Data Structure",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/design-add-and-search-words-data-structure/",
          "description": "Design a data structure that supports adding new words and searching for strings in data structure. Special characters '.' can match any single letter.",
          "details": {
            "key_idea": "To design a data structure that supports adding and searching words, we can use a Trie (prefix tree) with a special character '.' to represent any character. When searching, we traverse the Trie and recursively search in all child nodes for matching characters or '.'.",
            "time_complexity": "Insertion: O(m), Search: O(m)",
            "space_complexity": "O(n * m)",
            "python_solution": "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\n\nclass WordDictionary:\n    def __init__(self):\n        self.root = TrieNode()\n\n    def addWord(self, word: str) -> None:\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.is_end = True\n\n    def search(self, word: str) -> bool:\n        def search_in_node(node, word):\n            for i, char in enumerate(word):\n                if char not in node.children:\n                    if char == \".\":\n                        for child in node.children:\n                            if search_in_node(node.children[child], word[i + 1 :]):\n                                return True\n                    return False\n                else:\n                    node = node.children[char]\n            return node.is_end\n\n        return search_in_node(self.root, word)\n\n\n# Your WordDictionary object will be instantiated and called as such:\n# obj = WordDictionary()\n# obj.addWord(word)\n# param_2 = obj.search(word)\n"
          }
        },
        {
          "id": 212,
          "title": "Word Search II",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/word-search-ii/",
          "description": "Given an m x n grid of characters board and a string array words, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.",
          "details": {
            "key_idea": "To find all the words from a given list that can be formed by a 2D board of characters, we can use a Trie (prefix tree) to efficiently search for words while traversing the board. We start by building a Trie from the given list of words. Then, we perform a depth-first search (DFS) on the board, checking if the current path forms a valid prefix in the Trie. If it does, we continue the DFS until we find words or reach dead ends.",
            "time_complexity": "Building Trie: O(n * m), DFS: O(n * m * 4^k)",
            "space_complexity": "O(n * m)",
            "python_solution": "from collections import Counter\nfrom itertools import chain, product\nfrom typing import List\n\n\nclass TrieNode:\n    def __init__(self):\n        self.children = {}  # Store child nodes for each character\n        self.refcnt = 0  # Count of references to this node\n        self.is_word = False  # Flag to indicate if a complete word ends at this node\n        self.is_rev = False  # Flag to indicate if a word should be reversed\n\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()  # Initialize the root of the trie\n\n    def insert(self, word, rev):\n        node = self.root\n        for c in word:\n            node = node.children.setdefault(c, TrieNode())\n            node.refcnt += 1\n        node.is_word = True\n        node.is_rev = rev\n\n    def remove(self, word):\n        node = self.root\n        for i, c in enumerate(word):\n            parent = node\n            node = node.children[c]\n\n            if node.refcnt == 1:\n                path = [(parent, c)]\n                for c in word[i + 1 :]:\n                    path.append((node, c))\n                    node = node.children[c]\n                for parent, c in path:\n                    parent.children.pop(c)\n                return\n            node.refcnt -= 1\n        node.is_word = False\n\n\nclass Solution:\n    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:\n        res = []\n        n, m = len(board), len(board[0])\n        trie = Trie()\n\n        # Count characters on the board\n        boardcnt = Counter(chain(*board))\n\n        # Insert words into trie with appropriate orientation\n        for w, wrdcnt in ((w, Counter(w)) for w in words):\n            if any(wrdcnt[c] > boardcnt[c] for c in wrdcnt):\n                continue  # Skip if the word cannot be formed from the board\n            if wrdcnt[w[0]] < wrdcnt[w[-1]]:\n                trie.insert(w, False)\n            else:\n                trie.insert(w[::-1], True)\n\n        def dfs(r, c, parent) -> None:\n            if not (node := parent.children.get(board[r][c])):\n                return\n            path.append(board[r][c])\n            board[r][c] = \"#\"  # Mark visited cell\n\n            if node.is_word:\n                word = \"\".join(path)\n                res.append(word[::-1] if node.is_rev else word)\n                trie.remove(word)\n\n            # Explore neighboring cells\n            if r > 0:\n                dfs(r - 1, c, node)\n            if r < n - 1:\n                dfs(r + 1, c, node)\n            if c > 0:\n                dfs(r, c - 1, node)\n            if c < m - 1:\n                dfs(r, c + 1, node)\n\n            board[r][c] = path.pop()  # Backtrack and unmark cell\n\n        path = []\n        for r, c in product(range(n), range(m)):\n            dfs(r, c, trie.root)\n        return res"
          }
        }
      ]
    },
    {
      "topic_name": "Heap / Priority Queue",
      "problems": [
        {
          "id": 703,
          "title": "Kth Largest Element in a Stream",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/kth-largest-element-in-a-stream/",
          "description": "Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.",
          "details": {
            "key_idea": "To find the kth largest element in a stream of integers, we can use a min-heap (priority queue) with a maximum size of k. As new elements arrive, we add them to the min-heap. If the size of the heap exceeds k, we remove the smallest element. The top element of the min-heap will be the kth largest element.",
            "time_complexity": "Add: O(log k), Find Median: O(1)",
            "space_complexity": "O(k)",
            "python_solution": "import heapq\n\n\nclass KthLargest:\n    def __init__(self, k: int, nums: List[int]):\n        self.min_heap = []\n        self.k = k\n\n        for num in nums:\n            self.add(num)\n\n    def add(self, val: int) -> int:\n        heapq.heappush(self.min_heap, val)\n\n        if len(self.min_heap) > self.k:\n            heapq.heappop(self.min_heap)\n\n        return self.min_heap[0]"
          }
        },
        {
          "id": 1046,
          "title": "Last Stone Weight",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/last-stone-weight/",
          "description": "You are given an array of integers stones where stones[i] is the weight of the ith stone. We are playing a game with the stones. On each turn, we choose the two heaviest stones and smash them together. Suppose the heaviest stone has weight x and the second heaviest stone has weight y with x <= y. The result of the smash is a stone of weight y - x. Note that if x == y, both stones are destroyed. Return the weight of the last remaining stone. If there are no stones left, return 0.",
          "details": {
            "key_idea": "To simulate the process of smashing stones, we can use a max-heap (priority queue) to keep track of the stone weights. At each step, we pop the two largest stones from the heap, smash them together, and push the resulting weight back into the heap. We repeat this process until there is only one stone left in the heap, which will be the last stone weight.",
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "python_solution": "import heapq\n\n\nclass Solution:\n    def lastStoneWeight(self, stones: List[int]) -> int:\n        max_heap = [-stone for stone in stones]  # Use negative values for max-heap\n\n        heapq.heapify(max_heap)\n\n        while len(max_heap) > 1:\n            x = -heapq.heappop(max_heap)  # Extract the largest stone\n            y = -heapq.heappop(max_heap)  # Extract the second largest stone\n\n            if x != y:\n                heapq.heappush(max_heap, -(x - y))  # Push the remaining weight\n\n        return -max_heap[0] if max_heap else 0"
          }
        },
        {
          "id": 973,
          "title": "K Closest Points to Origin",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/k-closest-points-to-origin/",
          "description": "Given an array of points where points[i] = [xi, yi] represents the coordinates of the ith point, and an integer k, return the k closest points to the origin (0, 0). The distance between two points on the X-Y plane is the Euclidean distance (i.e., sqrt((x1 - x2)^2 + (y1 - y2)^2)).",
          "details": {
            "key_idea": "To find the k closest points to the origin, we can calculate the distance of each point from the origin and use a min-heap (priority queue) to keep track of the k closest points. As we iterate through the points, we push each point into the min-heap. If the size of the heap exceeds k, we remove the farthest point. The remaining points in the heap will be the k closest points.",
            "time_complexity": "O(n log k)",
            "space_complexity": "O(k)",
            "python_solution": "import heapq\n\n\nclass Solution:\n    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:\n        def distance(point):\n            return point[0] ** 2 + point[1] ** 2\n\n        min_heap = [(distance(point), point) for point in points]\n        heapq.heapify(min_heap)\n\n        result = []\n        for _ in range(k):\n            result.append(heapq.heappop(min_heap)[1])\n\n        return result"
          }
        },
        {
          "id": 215,
          "title": "Kth Largest Element in an Array",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/kth-largest-element-in-an-array/",
          "description": "Given an integer array nums and an integer k, return the kth largest element in the array. Note that it is the kth largest element in the sorted order, not the kth distinct element.",
          "details": {
            "key_idea": "To find the kth largest element in an array, we can use a min-heap (priority queue) with a maximum size of k. As we iterate through the array, we push elements into the min-heap. If the size of the heap exceeds k, we remove the smallest element. The top element of the min-heap will be the kth largest element.",
            "time_complexity": "O(n log k)",
            "space_complexity": "O(k)",
            "python_solution": "import heapq\n\n\nclass Solution:\n    def findKthLargest(self, nums: List[int], k: int) -> int:\n        min_heap = []\n\n        for num in nums:\n            heapq.heappush(min_heap, num)\n            if len(min_heap) > k:\n                heapq.heappop(min_heap)\n\n        return min_heap[0]"
          }
        },
        {
          "id": 621,
          "title": "Task Scheduler",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/task-scheduler/",
          "description": "Given a characters array tasks, representing the tasks you need to do, where tasks[i] is the ith task. Each task is an uppercase English letter representing a different task. Tasks could be done in any order. Each task takes one unit of time. For each unit of time, you can either perform one task or just idle. However, you have a cooling period n between two same tasks (i.e., if you perform the ith task at time t, then the next time you can perform the same task is at time t + n + 1). Return the minimum number of units of times that the tasks must be completed.",
          "details": {
            "key_idea": "To schedule tasks with maximum cooling time, we can use a greedy approach. We first count the frequency of each task and sort them in descending order. We then iterate through the tasks and use a cooldown counter to keep track of the remaining time until the next valid task can be scheduled. During each iteration, we schedule the task with the highest frequency that is not on cooldown. If there are no available tasks, we simply wait.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "import heapq\nfrom collections import Counter\n\n\nclass Solution:\n    def leastInterval(self, tasks: List[str], n: int) -> int:\n        task_counts = Counter(tasks)\n        max_heap = [-count for count in task_counts.values()]\n        heapq.heapify(max_heap)\n\n        cooldown = 0\n        while max_heap:\n            temp = []\n            for _ in range(n + 1):\n                if max_heap:\n                    temp.append(heapq.heappop(max_heap) + 1)\n\n            for count in temp:\n                if count < 0:\n                    heapq.heappush(max_heap, count)\n\n            if max_heap:\n                cooldown += n + 1\n            else:\n                cooldown += len(temp)\n\n        return cooldown"
          }
        },
        {
          "id": 355,
          "title": "Design Twitter",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/design-twitter/",
          "description": "Design a simplified version of Twitter where you can post tweets, follow/unfollow users, and get the most recent 10 tweets in the user's news feed.",
          "details": {
            "key_idea": "To design a simplified version of Twitter, we can use a combination of data structures. We maintain a dictionary to map users to their tweets, and a set of followees for each user. When a user posts a tweet, we add it to their tweet list. When a user wants to retrieve their news feed, we merge their own tweets and the tweets of their followees, and then sort them based on timestamps.",
            "time_complexity": "Post Tweet: O(1), Get News Feed: O(f + t log t)",
            "space_complexity": "O(u + t)",
            "python_solution": "import heapq\n\n\nclass Tweet:\n    def __init__(self, tweet_id, timestamp):\n        self.tweet_id = tweet_id\n        self.timestamp = timestamp\n\n\nclass Twitter:\n    def __init__(self):\n        self.user_tweets = {}  # User ID -> List of Tweet objects\n        self.user_followees = {}  # User ID -> Set of followees\n        self.timestamp = 0\n\n    def postTweet(self, userId: int, tweetId: int) -> None:\n        self.timestamp += 1\n        if userId not in self.user_tweets:\n            self.user_tweets[userId] = []\n        self.user_tweets[userId].append(Tweet(tweetId, self.timestamp))\n\n    def getNewsFeed(self, userId: int) -> List[int]:\n        tweets = []\n\n        if userId in self.user_tweets:\n            tweets.extend(self.user_tweets[userId])\n\n        if userId in self.user_followees:\n            for followee in self.user_followees[userId]:\n                if followee in self.user_tweets:\n                    tweets.extend(self.user_tweets[followee])\n\n        tweets.sort(key=lambda x: x.timestamp, reverse=True)\n        return [tweet.tweet_id for tweet in tweets[:10]]\n\n    def follow(self, followerId: int, followeeId: int) -> None:\n        if followerId != followeeId:\n            if followerId not in self.user_followees:\n                self.user_followees[followerId] = set()\n            self.user_followees[followerId].add(followeeId)\n\n    def unfollow(self, followerId: int, followeeId: int) -> None:\n        if (\n            followerId in self.user_followees\n            and followeeId in self.user_followees[followerId]\n        ):\n            self.user_followees[followerId].remove(followeeId)\n\n\n# Your Twitter object will be instantiated and called as such:\n# obj = Twitter()\n# obj.postTweet(userId,tweetId)\n# param_2 = obj.getNewsFeed(userId)\n# obj.follow(followerId,followeeId)\n# obj.unfollow(followerId,followeeId)"
          }
        },
        {
          "id": 295,
          "title": "Find Median from Data Stream",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/find-median-from-data-stream/",
          "description": "The median is the middle value in an ordered integer list, where if the size of the list is even, there is no middle value. If the size of the list is even, the median is the average of the two middle values. For example, median of [2,3] is (2 + 3) / 2 = 2.5. Given an integer array nums sorted in ascending order, return the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).",
          "details": {
            "key_idea": "To find the median from a stream of data, we can use two heaps: a max-heap to store the lower half of the data and a min-heap to store the upper half of the data. The max-heap will contain the smaller elements, and the min-heap will contain the larger elements. To maintain the balance of the heaps, we ensure that the size difference between the two heaps is at most 1. The median will be the average of the top elements of both heaps if the total number of elements is even, or the top element of the larger heap if the total number of elements is odd.",
            "time_complexity": "Add: O(log n), Find Median: O(1)",
            "space_complexity": "O(n)",
            "python_solution": "import heapq\n\n\nclass MedianFinder:\n    def __init__(self):\n        self.min_heap = []  # To store larger elements\n        self.max_heap = []  # To store smaller elements\n\n    def addNum(self, num: int) -> None:\n        if not self.max_heap or num <= -self.max_heap[0]:\n            heapq.heappush(self.max_heap, -num)\n        else:\n            heapq.heappush(self.min_heap, num)\n\n        # Balance the heaps\n        if len(self.max_heap) > len(self.min_heap) + 1:\n            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))\n        elif len(self.min_heap) > len(self.max_heap):\n            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))\n\n    def findMedian(self) -> float:\n        if len(self.max_heap) == len(self.min_heap):\n            return (-self.max_heap[0] + self.min_heap[0]) / 2\n        else:\n            return -self.max_heap[0]"
          }
        }
      ]
    },
    {
      "topic_name": "Backtracking",
      "problems": [
        {
          "id": 78,
          "title": "Subsets",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/subsets/",
          "description": "Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.",
          "details": {
            "key_idea": "To generate all possible subsets of a given set of distinct integers, we can use a recursive approach. For each element, we have two choices: either include it in the current subset or exclude it. We explore both choices recursively to generate all subsets.",
            "time_complexity": "O(2^n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def subsets(self, nums: List[int]) -> List[List[int]]:\n        def backtrack(start, subset):\n            subsets.append(subset[:])  # Append a copy of the current subset\n\n            for i in range(start, len(nums)):\n                subset.append(nums[i])\n                backtrack(i + 1, subset)\n                subset.pop()  # Backtrack\n\n        subsets = []\n        backtrack(0, [])\n        return subsets"
          }
        },
        {
          "id": 39,
          "title": "Combination Sum",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/combination-sum/",
          "description": "Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order. The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.",
          "details": {
            "key_idea": "To find all unique combinations that sum up to a target value, we can use a backtracking approach. Starting from each candidate element, we explore all possible combinations by adding the element to the current combination and recursively searching for the remaining sum. If the sum becomes equal to the target, we add the current combination to the result. This process is repeated for each candidate element.",
            "time_complexity": "O(k * 2^n)",
            "space_complexity": "O(target)",
            "python_solution": "class Solution:\n    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:\n        def backtrack(start, target, combination):\n            if target == 0:\n                result.append(combination[:])  # Append a copy of the current combination\n                return\n\n            for i in range(start, len(candidates)):\n                if candidates[i] > target:\n                    continue  # Skip if the candidate is too large\n\n                combination.append(candidates[i])\n                backtrack(i, target - candidates[i], combination)\n                combination.pop()  # Backtrack\n\n        result = []\n        backtrack(0, target, [])\n        return result"
          }
        },
        {
          "id": 46,
          "title": "Permutations",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/permutations/",
          "description": "Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.",
          "details": {
            "key_idea": "To generate all permutations of a given list of distinct integers, we can use a backtracking approach. Starting from each element, we explore all possible permutations by swapping the current element with other elements and recursively generating permutations for the remaining elements. Once we reach the end of the list, we add the current permutation to the result.",
            "time_complexity": "O(n!)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def permute(self, nums: List[int]) -> List[List[int]]:\n        def backtrack(start):\n            if start == len(nums) - 1:\n                permutations.append(nums[:])  # Append a copy of the current permutation\n\n            for i in range(start, len(nums)):\n                nums[start], nums[i] = nums[i], nums[start]  # Swap elements\n                backtrack(start + 1)\n                nums[start], nums[i] = nums[i], nums[start]  # Backtrack\n\n        permutations = []\n        backtrack(0)\n        return permutations"
          }
        },
        {
          "id": 90,
          "title": "Subsets II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/subsets-ii/",
          "description": "Given an integer array nums that may contain duplicates, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.",
          "details": {
            "key_idea": "To generate all possible subsets of a given list of integers, accounting for duplicates, we can use a backtracking approach. Similar to the previous subset problem, we explore all possible choices for each element: either include it in the current subset or exclude it. To handle duplicates, we skip adding the same element if it has already been processed at the same depth level.",
            "time_complexity": "O(2^n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:\n        def backtrack(start, subset):\n            subsets.append(subset[:])  # Append a copy of the current subset\n\n            for i in range(start, len(nums)):\n                if i > start and nums[i] == nums[i - 1]:\n                    continue  # Skip duplicates at the same depth level\n                subset.append(nums[i])\n                backtrack(i + 1, subset)\n                subset.pop()  # Backtrack\n\n        nums.sort()  # Sort the input to handle duplicates\n        subsets = []\n        backtrack(0, [])\n        return subsets"
          }
        },
        {
          "id": 40,
          "title": "Combination Sum II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/combination-sum-ii/",
          "description": "Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations of candidates where the chosen numbers sum to target. Each number in candidates may only be used once in the combination.",
          "details": {
            "key_idea": "To find all unique combinations that sum up to a target value, accounting for duplicates, we can use a backtracking approach. Starting from each candidate element, we explore all possible combinations by adding the element to the current combination and recursively searching for the remaining sum. To handle duplicates, we skip adding the same element if it has already been processed at the same depth level.",
            "time_complexity": "O(2^n)",
            "space_complexity": "O(target)",
            "python_solution": "class Solution:\n    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:\n        def backtrack(start, target, combination):\n            if target == 0:\n                result.append(combination[:])  # Append a copy of the current combination\n                return\n\n            for i in range(start, len(candidates)):\n                if i > start and candidates[i] == candidates[i - 1]:\n                    continue  # Skip duplicates at the same depth level\n\n                if candidates[i] > target:\n                    continue  # Skip if the candidate is too large\n\n                combination.append(candidates[i])\n                backtrack(i + 1, target - candidates[i], combination)\n                combination.pop()  # Backtrack\n\n        candidates.sort()  # Sort the input to handle duplicates\n        result = []\n        backtrack(0, target, [])\n        return result"
          }
        },
        {
          "id": 79,
          "title": "Word Search",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/word-search/",
          "description": "Given an m x n grid of characters board and a string word, return true if word exists in the grid. The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once. If the word exists in the grid, return true.",
          "details": {
            "key_idea": "To determine if a given word exists in the 2D board, we can perform a depth-first search (DFS) from each cell in the board. At each cell, we check if the current character of the word matches the character in the cell. If it does, we mark the cell as visited, recursively search its neighboring cells, and backtrack if the search is unsuccessful. We repeat this process for all cells in the board.",
            "time_complexity": "O(m * n * 4^k)",
            "space_complexity": "O(k)",
            "python_solution": "class Solution:\n    def exist(self, board: List[List[str]], word: str) -> bool:\n        def dfs(row, col, index):\n            if index == len(word):\n                return True\n\n            if (\n                row < 0\n                or row >= len(board)\n                or col < 0\n                or col >= len(board[0])\n                or board[row][col] != word[index]\n            ):\n                return False\n\n            original_char = board[row][col]\n            board[row][col] = \"#\"  # Mark the cell as visited\n\n            found = (\n                dfs(row + 1, col, index + 1)\n                or dfs(row - 1, col, index + 1)\n                or dfs(row, col + 1, index + 1)\n                or dfs(row, col - 1, index + 1)\n            )\n\n            board[row][col] = original_char  # Backtrack\n\n            return found\n\n        for row in range(len(board)):\n            for col in range(len(board[0])):\n                if board[row][col] == word[0] and dfs(row, col, 0):\n                    return True\n\n        return False"
          }
        },
        {
          "id": 131,
          "title": "Palindrome Partitioning",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/palindrome-partitioning/",
          "description": "Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.",
          "details": {
            "key_idea": "To partition a given string into palindromic substrings, we can use backtracking. Starting from each position in the string, we check if the substring from that position to the end is a palindrome. If it is, we recursively partition the remaining substring and continue the process. We keep track of the current partition in a list and store valid partitions in the result.",
            "time_complexity": "O(2^n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def partition(self, s: str) -> List[List[str]]:\n        def is_palindrome(sub):\n            return sub == sub[::-1]\n\n        def backtrack(start, partition):\n            if start == len(s):\n                result.append(partition[:])  # Append a copy of the current partition\n                return\n\n            for end in range(start + 1, len(s) + 1):\n                sub = s[start:end]\n                if is_palindrome(sub):\n                    partition.append(sub)\n                    backtrack(end, partition)\n                    partition.pop()  # Backtrack\n\n        result = []\n        backtrack(0, [])\n        return result"
          }
        },
        {
          "id": 17,
          "title": "Letter Combinations of a Phone Number",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/letter-combinations-of-a-phone-number/",
          "description": "Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order. A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.",
          "details": {
            "key_idea": "To generate all possible letter combinations of a phone number, we can use a recursive approach. Starting from each digit of the phone number, we generate combinations by appending each letter corresponding to the digit to the current combinations. We repeat this process for all digits and all possible letters, building up the combinations.",
            "time_complexity": "O(4^n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def letterCombinations(self, digits: str) -> List[str]:\n        if not digits:\n            return []\n\n        phone_mapping = {\n            \"2\": \"abc\",\n            \"3\": \"def\",\n            \"4\": \"ghi\",\n            \"5\": \"jkl\",\n            \"6\": \"mno\",\n            \"7\": \"pqrs\",\n            \"8\": \"tuv\",\n            \"9\": \"wxyz\",\n        }\n\n        def backtrack(index, combination):\n            if index == len(digits):\n                combinations.append(combination)\n                return\n\n            digit = digits[index]\n            letters = phone_mapping[digit]\n\n            for letter in letters:\n                backtrack(index + 1, combination + letter)\n\n        combinations = []\n        backtrack(0, \"\")\n        return combinations"
          }
        },
        {
          "id": 51,
          "title": "N-Queens",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/n-queens/",
          "description": "The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.",
          "details": {
            "key_idea": "To solve the N-Queens problem, we can use backtracking. Starting from each row, we try placing a queen in each column of that row and recursively move on to the next row. If a valid placement is found, we continue the process. We keep track of the board state and the positions of the queens to avoid conflicts.",
            "time_complexity": "O(N!)",
            "space_complexity": "O(N^2)",
            "python_solution": "class Solution:\n    def solveNQueens(self, n: int) -> List[List[str]]:\n        def is_safe(row, col):\n            # Check for conflicts with previous rows\n            for prev_row in range(row):\n                if board[prev_row][col] == \"Q\":\n                    return False\n                if (\n                    col - (row - prev_row) >= 0\n                    and board[prev_row][col - (row - prev_row)] == \"Q\"\n                ):\n                    return False\n                if (\n                    col + (row - prev_row) < n\n                    and board[prev_row][col + (row - prev_row)] == \"Q\"\n                ):\n                    return False\n            return True\n\n        def place_queen(row):\n            if row == n:\n                result.append([ \".\".join(row) for row in board])\n                return\n\n            for col in range(n):\n                if is_safe(row, col):\n                    board[row][col] = \"Q\"\n                    place_queen(row + 1)\n                    board[row][col] = \".\"\n\n        board = [[\".\" for _ in range(n)] for _ in range(n)]\n        result = []\n        place_queen(0)\n        return result"
          }
        }
      ]
    },
    {
      "topic_name": "Graphs",
      "problems": [
        {
          "id": 200,
          "title": "Number of Islands",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/number-of-islands/",
          "description": "Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.",
          "details": {
            "key_idea": "The problem is to count the number of islands in a 2D grid where '1' represents land and '0' represents water. We can solve this problem using Depth-First Search (DFS) algorithm. For each cell that contains '1', we perform DFS to explore all adjacent land cells and mark them as visited by changing their value to '0'. This way, we count each connected component of '1's as a separate island.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def numIslands(self, grid: List[List[str]]) -> int:\n        if not grid:\n            return 0\n\n        rows, cols = len(grid), len(grid[0])\n        count = 0\n\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(grid)\n                or col < 0\n                or col >= len(grid[0])\n                or grid[row][col] == \"0\"\n            ):\n                return\n\n            grid[row][col] = \"0\"  # Mark the cell as visited\n            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]\n\n            for dr, dc in directions:\n                dfs(row + dr, col + dc)\n\n        for i in range(rows):\n            for j in range(cols):\n                if grid[i][j] == \"1\":\n                    count += 1\n                    dfs(i, j)\n\n        return count"
          }
        },
        {
          "id": 133,
          "title": "Clone Graph",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/clone-graph/",
          "description": "Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.",
          "details": {
            "key_idea": "The problem is to clone an undirected graph. We can solve this using Depth-First Search (DFS) or Breadth-First Search (BFS). Here, I'm use DFS to traverse the original graph and create a new graph.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "python_solution": "# Definition for a Node.\n# class Node:\n#     def __init__(self, val = 0, neighbors = None):\n#         self.val = val\n#         self.neighbors = neighbors if neighbors is not None else []\n\n\nclass Solution:\n    def cloneGraph(self, node: \"Node\") -> \"Node\":\n        if not node:\n            return None\n\n        visited = {}  # Dictionary to store the cloned nodes\n\n        def dfs(original_node):\n            if original_node in visited:\n                return visited[original_node]\n\n            new_node = Node(original_node.val)\n            visited[original_node] = new_node\n\n            for neighbor in original_node.neighbors:\n                new_neighbor = dfs(neighbor)\n                new_node.neighbors.append(new_neighbor)\n\n            return new_node\n\n        return dfs(node)"
          }
        },
        {
          "id": 695,
          "title": "Max Area of Island",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/max-area-of-island/",
          "description": "You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water. The area of an island is the number of cells with a value 1 in the island. Return the maximum area of an island in grid. If there is no island, return 0.",
          "details": {
            "key_idea": "The problem is to find the maximum area of an island in a grid where 1 represents land and 0 represents water. We can solve this using Depth-First Search (DFS) to traverse each cell of the grid and identify connected land cells forming an island.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(grid)\n                or col < 0\n                or col >= len(grid[0])\n                or grid[row][col] == 0\n            ):\n                return 0\n\n            grid[row][col] = 0  # Mark as visited\n            area = 1\n\n            area += dfs(row + 1, col)  # Check down\n            area += dfs(row - 1, col)  # Check up\n            area += dfs(row, col + 1)  # Check right\n            area += dfs(row, col - 1)  # Check left\n\n            return area\n\n        max_area = 0\n        for row in range(len(grid)):\n            for col in range(len(grid[0])):\n                if grid[row][col] == 1:\n                    max_area = max(max_area, dfs(row, col))\n\n        return max_area"
          }
        },
        {
          "id": 417,
          "title": "Pacific Atlantic Water Flow",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/pacific-atlantic-water-flow/",
          "description": "There is an m x n rectangular matrix given where each element is the height of the integer. You are given an integer matrix heights representing the height of each unit cell in a grid. You are also given two integers rows and cols representing the number of cells in the grid. You can move from a cell to an adjacent cell in the 4 directions (up, down, left, right) if the height of the adjacent cell is less than or equal to the height of the current cell. Water can flow to the Pacific ocean if it can reach the ocean from the current cell. Water can flow to the Atlantic ocean if it can reach the ocean from the current cell. Return a list of grid coordinates where water can flow to both the Pacific and Atlantic oceans.",
          "details": {
            "key_idea": "The problem is to find the cells in a matrix where water can flow from both the Pacific Ocean and the Atlantic Ocean. We can solve this using Depth-First Search (DFS) starting from the ocean borders. Each cell that can be reached from both oceans will be added to the final result.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:\n        if not heights:\n            return []\n\n        rows, cols = len(heights), len(heights[0])\n        pacific_reachable = set()\n        atlantic_reachable = set()\n\n        def dfs(r, c, reachable):\n            reachable.add((r, c))\n            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:\n                nr, nc = r + dr, c + dc\n                if (\n                    0 <= nr < rows\n                    and 0 <= nc < cols\n                    and (nr, nc) not in reachable\n                    and heights[nr][nc] >= heights[r][c]\n                ):\n                    dfs(nr, nc, reachable)\n\n        for r in range(rows):\n            dfs(r, 0, pacific_reachable)\n            dfs(r, cols - 1, atlantic_reachable)\n\n        for c in range(cols):\n            dfs(0, c, pacific_reachable)\n            dfs(rows - 1, c, atlantic_reachable)\n\n        return list(pacific_reachable & atlantic_reachable)"
          }
        },
        {
          "id": 130,
          "title": "Surrounded Regions",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/surrounded-regions/",
          "description": "Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'. A region is captured by flipping all 'O's into 'X's in that surrounded region. An 'O' cell is considered surrounded if it is not on the border and not connected to any 'O' cell on the border.",
          "details": {
            "key_idea": "The problem is to capture 'O' cells that are not surrounded by 'X' cells in a given board. To solve this, we can use Depth-First Search (DFS) starting from the boundary 'O' cells. All the 'O' cells that are reachable from the boundary will be retained, and the rest will be changed to 'X'.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def solve(self, board: List[List[str]]) -> None:\n        \"\"\"\n        Do not return anything, modify board in-place instead.\n        \"\"\"\n\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(board)\n                or col < 0\n                or col >= len(board[0])\n                or board[row][col] != \"O\"\n            ):\n                return\n\n            board[row][col] = \"E\"  # Mark as visited but not surrounded\n\n            # Check adjacent cells\n            dfs(row + 1, col)  # Check down\n            dfs(row - 1, col)  # Check up\n            dfs(row, col + 1)  # Check right\n            dfs(row, col - 1)  # Check left\n\n        # Traverse the boundary and mark connected 'O' cells as 'E'\n        for row in range(len(board)):\n            if board[row][0] == \"O\":\n                dfs(row, 0)\n            if board[row][len(board[0]) - 1] == \"O\":\n                dfs(row, len(board[0]) - 1)\n\n        for col in range(len(board[0])):\n            if board[0][col] == \"O\":\n                dfs(0, col)\n            if board[len(board) - 1][col] == \"O\":\n                dfs(len(board) - 1, col)\n\n        # Mark internal 'O' cells as 'X' and restore 'E' cells to 'O'\n        for row in range(len(board)):\n            for col in range(len(board[0])):\n                if board[row][col] == \"O\":\n                    board[row][col] = \"X\"\n                elif board[row][col] == \"E\":\n                    board[row][col] = \"O\""
          }
        },
        {
          "id": 994,
          "title": "Rotting Oranges",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/rotting-oranges/",
          "description": "You are given an m x n grid of oranges where oranges[i][j] can be: 0: empty cell, 1: fresh orange, 2: rotten orange. Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten. Return the minimum number of minutes that must elapse until no fresh orange remains. If this is impossible, return -1.",
          "details": {
            "key_idea": "The problem is to determine the minimum time needed for all oranges to become rotten, considering that rotten oranges can also infect adjacent fresh oranges in each minute. We can model this problem using Breadth-First Search (BFS), where each minute corresponds to a level of the BFS traversal.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "from collections import deque\n\n\nclass Solution:\n    def orangesRotting(self, grid: List[List[int]]) -> int:\n        if not grid:\n            return -1\n\n        rows, cols = len(grid), len(grid[0])\n        fresh_count = 0  # Count of fresh oranges\n        rotten = deque()  # Queue to store coordinates of rotten oranges\n        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible adjacent cells\n\n        # Initialize the queue with coordinates of rotten oranges\n        for row in range(rows):\n            for col in range(cols):\n                if grid[row][col] == 2:\n                    rotten.append((row, col))\n                elif grid[row][col] == 1:\n                    fresh_count += 1\n\n        minutes = 0  # Timer\n\n        while rotten:\n            level_size = len(rotten)\n\n            for _ in range(level_size):\n                row, col = rotten.popleft()\n\n                for dr, dc in directions:\n                    new_row, new_col = row + dr, col + dc\n\n                    # Check if the new cell is within bounds and has a fresh orange\n                    if (\n                        0 <= new_row < rows\n                        and 0 <= new_col < cols\n                        and grid[new_row][new_col] == 1\n                    ):\n                        grid[new_row][new_col] = 2  # Infect the fresh orange\n                        fresh_count -= 1\n                        rotten.append((new_row, new_col))\n\n            if rotten:\n                minutes += 1\n\n        # If there are fresh oranges left, return -1; otherwise, return the elapsed minutes\n        return minutes if fresh_count == 0 else -1"
          }
        },
        {
          "id": 286,
          "title": "Walls and Gates",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/walls-and-gates/",
          "description": "You are given a 2D grid representing a map of rooms where '0' represents a gate, '-1' represents a wall, and INF (represented by 2147483647) represents an empty room. Distances are measured via the number of steps the shortest path between different rooms. Distances are measured via the number of steps the shortest path between different rooms. Return the map of rooms filled with their distances to the nearest gate.",
          "details": {
            "key_idea": "The problem is to fill each empty room (represented by INF) with the distance to the nearest gate. This can be approached using Breadth-First Search (BFS), where the gates are the starting points and the empty rooms are visited layer by layer.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def wallsAndGates(self, rooms: List[List[int]]) -> None:\n        if not rooms:\n            return\n\n        rows, cols = len(rooms), len(rooms[0])\n        gates = [(i, j) for i in range(rows) for j in range(cols) if rooms[i][j] == 0]\n        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]\n\n        for gate_row, gate_col in gates:\n            visited = set()  # To track visited cells in BFS\n            queue = deque([(gate_row, gate_col, 0)])\n\n            while queue:\n                row, col, distance = queue.popleft()\n                rooms[row][col] = min(rooms[row][col], distance)\n                visited.add((row, col))\n\n                for dr, dc in directions:\n                    new_row, new_col = row + dr, col + dc\n\n                    if (\n                        0 <= new_row < rows\n                        and 0 <= new_col < cols\n                        and rooms[new_row][new_col] != -1\n                        and (new_row, new_col) not in visited\n                    ):\n                        queue.append((new_row, new_col, distance + 1))\n                        visited.add((new_row, new_col))"
          }
        },
        {
          "id": 207,
          "title": "Course Schedule",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/course-schedule/",
          "description": "There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. For example, if prerequisites = [[1,0],[2,0],[3,1],[3,2]], then you will need to take course 0 to take course 1 and course 2. You need to take course 1 or 2 to take course 3. Return true if you can finish all courses. Otherwise, return false.",
          "details": {
            "key_idea": "The problem can be reduced to detecting cycles in a directed graph. We can represent the course prerequisites as directed edges between nodes (courses). If there is a cycle in the graph, it means we can't complete all courses.",
            "time_complexity": "O(numCourses + len(prerequisites))",
            "space_complexity": "O(numCourses + len(prerequisites))",
            "python_solution": "class Solution:\n    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:\n        graph = {i: [] for i in range(numCourses)}\n        in_degree = [0] * numCourses\n\n        # Construct the graph and count in-degrees\n        for course, prereq in prerequisites:\n            graph[prereq].append(course)\n            in_degree[course] += 1\n\n        # Initialize a queue with nodes having in-degree zero\n        queue = collections.deque(\n            [course for course, degree in enumerate(in_degree) if degree == 0]\n        )\n\n        # Perform topological sorting and update in-degrees\n        while queue:\n            node = queue.popleft()\n            for neighbor in graph[node]:\n                in_degree[neighbor] -= 1\n                if in_degree[neighbor] == 0:\n                    queue.append(neighbor)\n\n        # If any course has in-degree greater than zero, there's a cycle\n        return all(degree == 0 for degree in in_degree)"
          }
        },
        {
          "id": 210,
          "title": "Course Schedule II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/course-schedule-ii/",
          "description": "There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. Return the ordering of courses you should take to finish all courses. If there are many valid orderings, return any of them. If it is impossible to finish all courses, return an empty array.",
          "details": {
            "key_idea": "This problem is an extension of the previous Course Schedule problem (LeetCode 207). We need to return the order in which courses can be taken. We can use the topological sorting approach to solve this.",
            "time_complexity": "O(numCourses + len(prerequisites))",
            "space_complexity": "O(numCourses + len(prerequisites))",
            "python_solution": "class Solution:\n    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:\n        graph = {i: [] for i in range(numCourses)}\n        in_degree = [0] * numCourses\n        order = []\n\n        # Construct the graph and count in-degrees\n        for course, prereq in prerequisites:\n            graph[prereq].append(course)\n            in_degree[course] += 1\n\n        # Initialize a queue with nodes having in-degree zero\n        queue = collections.deque(\n            [course for course, degree in enumerate(in_degree) if degree == 0]\n        )\n\n        # Perform topological sorting and update in-degrees\n        while queue:\n            node = queue.popleft()\n            order.append(node)\n            for neighbor in graph[node]:\n                in_degree[neighbor] -= 1\n                if in_degree[neighbor] == 0:\n                    queue.append(neighbor)\n\n        # If the order doesn't contain all courses, there's a cycle\n        return order if len(order) == numCourses else []"
          }
        },
        {
          "id": 684,
          "title": "Redundant Connection",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/redundant-connection/",
          "description": "In this problem, a tree is an undirected graph that is connected and has no cycles. You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge causes the graph to have a cycle. Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.",
          "details": {
            "key_idea": "This problem can be solved using the Union-Find (Disjoint Set Union) algorithm. We initialize each node as its own parent and iterate through the given edges. For each edge, we check if the nodes have the same parent. If they do, that means adding this edge will create a cycle, and it's the redundant edge. If they don't have the same parent, we merge their sets by updating one's parent to be the other.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:\n        n = len(edges)\n        parent = list(range(n + 1))  # Initialize each node as its own parent\n\n        def find(x):\n            if parent[x] != x:\n                parent[x] = find(parent[x])  # Path compression\n            return parent[x]\n\n        def union(x, y):\n            parent[find(x)] = find(y)\n\n        for edge in edges:\n            u, v = edge\n            if find(u) == find(v):\n                return edge\n            union(u, v)\n\n        return []"
          }
        },
        {
          "id": 323,
          "title": "Number of Connected Components in An Undirected Graph",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/",
          "description": "You are given an integer n which represents the number of nodes in the graph. You are also given a 2D integer array edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi. Return the number of connected components in the graph.",
          "details": {
            "key_idea": "This problem can be solved using Depth-First Search (DFS) or Breadth-First Search (BFS). We represent the given edges as an adjacency list, where each node points to its neighboring nodes. We then iterate through all nodes and perform a DFS/BFS from each unvisited node to explore all connected components. The number of times we need to start a new DFS/BFS corresponds to the number of connected components in the graph.",
            "time_complexity": "O(n + m)",
            "space_complexity": "O(n + m)",
            "python_solution": "from collections import defaultdict, deque\n\n\nclass Solution:\n    def countComponents(self, n: int, edges: List[List[int]]) -> int:\n        graph = defaultdict(list)\n        for u, v in edges:\n            graph[u].append(v)\n            graph[v].append(u)\n\n        def dfs(node):\n            visited.add(node)\n            for neighbor in graph[node]:\n                if neighbor not in visited:\n                    dfs(neighbor)\n\n        visited = set()\n        components = 0\n\n        for node in range(n):\n            if node not in visited:\n                components += 1\n                dfs(node)\n\n        return components"
          }
        },
        {
          "id": 261,
          "title": "Graph Valid Tree",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/graph-valid-tree/",
          "description": "Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.",
          "details": {
            "key_idea": "This problem can be solved using Depth-First Search (DFS) or Union-Find algorithm. We represent the given edges as an adjacency list, where each node points to its neighboring nodes. To determine whether the graph is a valid tree, we need to check two conditions: 1. The graph must be connected, i.e., there is a path between every pair of nodes. 2. There should be no cycles in the graph.",
            "time_complexity": "O(n + m)",
            "space_complexity": "O(n + m)",
            "python_solution": "from collections import defaultdict, deque\n\n\nclass Solution:\n    def validTree(self, n: int, edges: List[List[int]]) -> bool:\n        if len(edges) != n - 1:\n            return False\n\n        graph = defaultdict(list)\n        for u, v in edges:\n            graph[u].append(v)\n            graph[v].append(u)\n\n        visited = set()\n\n        def dfs(node, parent):\n            visited.add(node)\n            for neighbor in graph[node]:\n                if neighbor != parent:\n                    if neighbor in visited or not dfs(neighbor, node):\n                        return False\n            return True\n\n        # Check if the graph is connected\n        if not dfs(0, -1):\n            return False\n\n        return len(visited) == n"
          }
        },
        {
          "id": 127,
          "title": "Word Ladder",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/word-ladder/",
          "description": "A transformation sequence from wordBegin to wordEnd using a dictionary wordList is a sequence of wordsbeginWord -> s1 -> s2 -> ... -> sk such that: Every adjacent pair of words in the sequence form a transformation. That is, only one letter can be changed at a time between two adjacent words. Every word in the sequence including beginWord and endWord is in wordList. Return the shortest transformation sequence's length in words, or 0 if no such sequence exists.",
          "details": {
            "key_idea": "This problem can be solved using a breadth-first search (BFS) approach. We start with the given beginWord and perform a BFS to explore all possible word transformations, one character change at a time. We maintain a queue to track the current word and its transformation path. For each word in the queue, we generate all possible words by changing one character at a time and check if it's in the word list. If it is, we add it to the queue and mark it as visited. We continue this process until we reach the endWord or the queue is empty.",
            "time_complexity": "O(n * m)",
            "space_complexity": "O(n)",
            "python_solution": "from collections import deque\n\n\nclass Solution:\n    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:\n        wordSet = set(wordList)\n        if endWord not in wordSet:\n            return 0\n\n        queue = deque([(beginWord, 1)])  # Start from the beginWord with level 1\n        visited = set()\n\n        while queue:\n            word, level = queue.popleft()\n            if word == endWord:\n                return level\n\n            for i in range(len(word)):\n                for c in \"abcdefghijklmnopqrstuvwxyz\":\n                    new_word = word[:i] + c + word[i + 1 :]\n                    if new_word in wordSet and new_word not in visited:\n                        visited.add(new_word)\n                        queue.append((new_word, level + 1))\n\n        return 0"
          }
        },
        {
          "id": 323,
          "title": "Number of Connected Components in An Undirected Graph",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/",
          "description": "You are given an integer n which represents the number of nodes in the graph. You are also given a 2D integer array edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi. Return the number of connected components in the graph.",
          "details": {
            "key_idea": "This problem can be solved using Depth-First Search (DFS) or Breadth-First Search (BFS). We represent the given edges as an adjacency list, where each node points to its neighboring nodes. We then iterate through all nodes and perform a DFS/BFS from each unvisited node to explore all connected components. The number of times we need to start a new DFS/BFS corresponds to the number of connected components in the graph.",
            "time_complexity": "O(n + m)",
            "space_complexity": "O(n + m)",
            "python_solution": "from collections import defaultdict, deque\n\n\nclass Solution:\n    def countComponents(self, n: int, edges: List[List[int]]) -> int:\n        graph = defaultdict(list)\n        for u, v in edges:\n            graph[u].append(v)\n            graph[v].append(u)\n\n        def dfs(node):\n            visited.add(node)\n            for neighbor in graph[node]:\n                if neighbor not in visited:\n                    dfs(neighbor)\n\n        visited = set()\n        components = 0\n\n        for node in range(n):\n            if node not in visited:\n                components += 1\n                dfs(node)\n\n        return components"
          }
        },
        {
          "id": 130,
          "title": "Surrounded Regions",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/surrounded-regions/",
          "description": "Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'. A region is captured by flipping all 'O's into 'X's in that surrounded region. A region is captured by flipping all 'O's into 'X's in that surrounded region. An 'O' cell is considered surrounded if it is not on the border and not connected to any 'O' cell on the border.",
          "details": {
            "key_idea": "The problem is to capture 'O' cells that are not surrounded by 'X' cells in a given board. To solve this, we can use Depth-First Search (DFS) starting from the boundary 'O' cells. All the 'O' cells that are reachable from the boundary will be retained, and the rest will be changed to 'X'.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def solve(self, board: List[List[str]]) -> None:\n        \"\"\"\n        Do not return anything, modify board in-place instead.\n        \"\"\"\n\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(board)\n                or col < 0\n                or col >= len(board[0])\n                or board[row][col] != \"O\"\n            ):\n                return\n\n            board[row][col] = \"E\"  # Mark as visited but not surrounded\n\n            # Check adjacent cells\n            dfs(row + 1, col)  # Check down\n            dfs(row - 1, col)  # Check up\n            dfs(row, col + 1)  # Check right\n            dfs(row, col - 1)  # Check left\n\n        # Traverse the boundary and mark connected 'O' cells as 'E'\n        for row in range(len(board)):\n            if board[row][0] == \"O\":\n                dfs(row, 0)\n            if board[row][len(board[0]) - 1] == \"O\":\n                dfs(row, len(board[0]) - 1)\n\n        for col in range(len(board[0])):\n            if board[0][col] == \"O\":\n                dfs(0, col)\n            if board[len(board) - 1][col] == \"O\":\n                dfs(len(board) - 1, col)\n\n        # Mark internal 'O' cells as 'X' and restore 'E' cells to 'O'\n        for row in range(len(board)):\n            for col in range(len(board[0])):\n                if board[row][col] == \"O\":\n                    board[row][col] = \"X\"\n                elif board[row][col] == \"E\":\n                    board[row][col] = \"O\""
          }
        },
        {
          "id": 695,
          "title": "Max Area of Island",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/max-area-of-island/",
          "description": "You are given an m x n 2D grid where grid[i][j] is either 1 (land) or 0 (water). An island is a group of 1's connected 4-directionally (horizontal or vertical). You may assume all four edges of the grid are surrounded by water. The area of an island is the number of cells with a value 1 in the island. Return the maximum area of an island in grid. If there is no island, return 0.",
          "details": {
            "key_idea": "The problem is to find the maximum area of an island in a grid where 1 represents land and 0 represents water. We can solve this using Depth-First Search (DFS) to traverse each cell of the grid and identify connected land cells forming an island.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(grid)\n                or col < 0\n                or col >= len(grid[0])\n                or grid[row][col] == 0\n            ):\n                return 0\n\n            grid[row][col] = 0  # Mark as visited\n            area = 1\n\n            area += dfs(row + 1, col)  # Check down\n            area += dfs(row - 1, col)  # Check up\n            area += dfs(row, col + 1)  # Check right\n            area += dfs(row, col - 1)  # Check left\n\n            return area\n\n        max_area = 0\n        for row in range(len(grid)):\n            for col in range(len(grid[0])):\n                if grid[row][col] == 1:\n                    max_area = max(max_area, dfs(row, col))\n\n        return max_area"
          }
        },
        {
          "id": 417,
          "title": "Pacific Atlantic Water Flow",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/pacific-atlantic-water-flow/",
          "description": "Given an m x n matrix of non-negative integers representing the height of each unit cell in a grid, and two integers rows and cols representing the number of cells in the grid. You can move to an adjacent cell if the height of the adjacent cell is less than or equal to the height of the current cell. Water can flow to the Pacific ocean if it can reach the ocean from the current cell. Water can flow to the Atlantic ocean if it can reach the ocean from the current cell. Return a list of grid coordinates where water can flow to both the Pacific and Atlantic oceans.",
          "details": {
            "key_idea": "The problem is to find the cells in a matrix where water can flow from both the Pacific Ocean and the Atlantic Ocean. We can solve this using Depth-First Search (DFS) starting from the ocean borders. Each cell that can be reached from both oceans will be added to the final result.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:\n        if not heights:\n            return []\n\n        rows, cols = len(heights), len(heights[0])\n        pacific_reachable = set()\n        atlantic_reachable = set()\n\n        def dfs(r, c, reachable):\n            reachable.add((r, c))\n            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:\n                nr, nc = r + dr, c + dc\n                if (\n                    0 <= nr < rows\n                    and 0 <= nc < cols\n                    and (nr, nc) not in reachable\n                    and heights[nr][nc] >= heights[r][c]\n                ):\n                    dfs(nr, nc, reachable)\n\n        for r in range(rows):\n            dfs(r, 0, pacific_reachable)\n            dfs(r, cols - 1, atlantic_reachable)\n\n        for c in range(cols):\n            dfs(0, c, pacific_reachable)\n            dfs(rows - 1, c, atlantic_reachable)\n\n        return list(pacific_reachable & atlantic_reachable)"
          }
        },
        {
          "id": 130,
          "title": "Surrounded Regions",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/surrounded-regions/",
          "description": "Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'. A region is captured by flipping all 'O's into 'X's in that surrounded region. An 'O' cell is considered surrounded if it is not on the border and not connected to any 'O' cell on the border.",
          "details": {
            "key_idea": "The problem is to capture 'O' cells that are not surrounded by 'X' cells in a given board. To solve this, we can use Depth-First Search (DFS) starting from the boundary 'O' cells. All the 'O' cells that are reachable from the boundary will be retained, and the rest will be changed to 'X'.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def solve(self, board: List[List[str]]) -> None:\n        \"\"\"\n        Do not return anything, modify board in-place instead.\n        \"\"\"\n\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(board)\n                or col < 0\n                or col >= len(board[0])\n                or board[row][col] != \"O\"\n            ):\n                return\n\n            board[row][col] = \"E\"  # Mark as visited but not surrounded\n\n            # Check adjacent cells\n            dfs(row + 1, col)  # Check down\n            dfs(row - 1, col)  # Check up\n            dfs(row, col + 1)  # Check right\n            dfs(row, col - 1)  # Check left\n\n        # Traverse the boundary and mark connected 'O' cells as 'E'\n        for row in range(len(board)):\n            if board[row][0] == \"O\":\n                dfs(row, 0)\n            if board[row][len(board[0]) - 1] == \"O\":\n                dfs(row, len(board[0]) - 1)\n\n        for col in range(len(board[0])):\n            if board[0][col] == \"O\":\n                dfs(0, col)\n            if board[len(board) - 1][col] == \"O\":\n                dfs(len(board) - 1, col)\n\n        # Mark internal 'O' cells as 'X' and restore 'E' cells to 'O'\n        for row in range(len(board)):\n            for col in range(len(board[0])):\n                if board[row][col] == \"O\":\n                    board[row][col] = \"X\"\n                elif board[row][col] == \"E\":\n                    board[row][col] = \"O\""
          }
        },
        {
          "id": 200,
          "title": "Number of Islands",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/number-of-islands/",
          "description": "Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.",
          "details": {
            "key_idea": "The problem is to count the number of islands in a 2D grid where '1' represents land and '0' represents water. We can solve this problem using Depth-First Search (DFS) algorithm. For each cell that contains '1', we perform DFS to explore all adjacent land cells and mark them as visited by changing their value to '0'. This way, we count each connected component of '1's as a separate island.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def numIslands(self, grid: List[List[str]]) -> int:\n        if not grid:\n            return 0\n\n        rows, cols = len(grid), len(grid[0])\n        count = 0\n\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= rows\n                or col < 0\n                or col >= cols\n                or grid[row][col] == \"0\"\n            ):\n                return\n\n            grid[row][col] = \"0\"  # Mark the cell as visited\n            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]\n\n            for dr, dc in directions:\n                dfs(row + dr, col + dc)\n\n        for i in range(rows):\n            for j in range(cols):\n                if grid[i][j] == \"1\":\n                    count += 1\n                    dfs(i, j)\n\n        return count"
          }
        },
        {
          "id": 133,
          "title": "Clone Graph",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/clone-graph/",
          "description": "Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.",
          "details": {
            "key_idea": "The problem is to clone an undirected graph. We can solve this using Depth-First Search (DFS) or Breadth-First Search (BFS). Here, I'm use DFS to traverse the original graph and create a new graph.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "python_solution": "# Definition for a Node.\n# class Node:\n#     def __init__(self, val = 0, neighbors = None):\n#         self.val = val\n#         self.neighbors = neighbors if neighbors is not None else []\n\n\nclass Solution:\n    def cloneGraph(self, node: \"Node\") -> \"Node\":\n        if not node:\n            return None\n\n        visited = {}  # Dictionary to store the cloned nodes\n\n        def dfs(original_node):\n            if original_node in visited:\n                return visited[original_node]\n\n            new_node = Node(original_node.val)\n            visited[original_node] = new_node\n\n            for neighbor in original_node.neighbors:\n                new_neighbor = dfs(neighbor)\n                new_node.neighbors.append(new_neighbor)\n\n            return new_node\n\n        return dfs(node)"
          }
        },
        {
          "id": 695,
          "title": "Max Area of Island",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/max-area-of-island/",
          "description": "You are given an m x n binary matrix grid where 1 represents land and 0 represents water. An island is a group of 1's connected 4-directionally (horizontal or vertical). You may assume all four edges of the grid are surrounded by water. The area of an island is the number of cells with a value 1 in the island. Return the maximum area of an island in grid. If there is no island, return 0.",
          "details": {
            "key_idea": "The problem is to find the maximum area of an island in a grid where 1 represents land and 0 represents water. We can solve this using Depth-First Search (DFS) to traverse each cell of the grid and identify connected land cells forming an island.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(grid)\n                or col < 0\n                or col >= len(grid[0])\n                or grid[row][col] == 0\n            ):\n                return 0\n\n            grid[row][col] = 0  # Mark as visited\n            area = 1\n\n            area += dfs(row + 1, col)  # Check down\n            area += dfs(row - 1, col)  # Check up\n            area += dfs(row, col + 1)  # Check right\n            area += dfs(row, col - 1)  # Check left\n\n            return area\n\n        max_area = 0\n        for row in range(len(grid)):\n            for col in range(len(grid[0])):\n                if grid[row][col] == 1:\n                    max_area = max(max_area, dfs(row, col))\n\n        return max_area"
          }
        },
        {
          "id": 417,
          "title": "Pacific Atlantic Water Flow",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/pacific-atlantic-water-flow/",
          "description": "There is an m x n rectangular matrix given where each element is the height of the integer. You are given an integer matrix heights representing the height of each unit cell in a grid. You are also given two integers rows and cols representing the number of cells in the grid. You can move from a cell to an adjacent cell in the 4 directions (up, down, left, right) if the height of the adjacent cell is less than or equal to the height of the current cell. Water can flow to the Pacific ocean if it can reach the ocean from the current cell. Water can flow to the Atlantic ocean if it can reach the ocean from the current cell. Return a list of grid coordinates where water can flow to both the Pacific and Atlantic oceans.",
          "details": {
            "key_idea": "The problem is to find the cells in a matrix where water can flow from both the Pacific Ocean and the Atlantic Ocean. We can solve this using Depth-First Search (DFS) starting from the ocean borders. Each cell that can be reached from both oceans will be added to the final result.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:\n        if not heights:\n            return []\n\n        rows, cols = len(heights), len(heights[0])\n        pacific_reachable = set()\n        atlantic_reachable = set()\n\n        def dfs(r, c, reachable):\n            reachable.add((r, c))\n            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:\n                nr, nc = r + dr, c + dc\n                if (\n                    0 <= nr < rows\n                    and 0 <= nc < cols\n                    and (nr, nc) not in reachable\n                    and heights[nr][nc] >= heights[r][c]\n                ):\n                    dfs(nr, nc, reachable)\n\n        for r in range(rows):\n            dfs(r, 0, pacific_reachable)\n            dfs(r, cols - 1, atlantic_reachable)\n\n        for c in range(cols):\n            dfs(0, c, pacific_reachable)\n            dfs(rows - 1, c, atlantic_reachable)\n\n        return list(pacific_reachable & atlantic_reachable)"
          }
        },
        {
          "id": 130,
          "title": "Surrounded Regions",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/surrounded-regions/",
          "description": "Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'. A region is captured by flipping all 'O's into 'X's in that surrounded region. An 'O' cell is considered surrounded if it is not on the border and not connected to any 'O' cell on the border.",
          "details": {
            "key_idea": "The problem is to capture 'O' cells that are not surrounded by 'X' cells in a given board. To solve this, we can use Depth-First Search (DFS) starting from the boundary 'O' cells. All the 'O' cells that are reachable from the boundary will be retained, and the rest will be changed to 'X'.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def solve(self, board: List[List[str]]) -> None:\n        \"\"\"\n        Do not return anything, modify board in-place instead.\n        \"\"\"\n\n        def dfs(row, col):\n            if (\n                row < 0\n                or row >= len(board)\n                or col < 0\n                or col >= len(board[0])\n                or board[row][col] != \"O\"\n            ):\n                return\n\n            board[row][col] = \"E\"  # Mark as visited but not surrounded\n\n            # Check adjacent cells\n            dfs(row + 1, col)  # Check down\n            dfs(row - 1, col)  # Check up\n            dfs(row, col + 1)  # Check right\n            dfs(row, col - 1)  # Check left\n\n        # Traverse the boundary and mark connected 'O' cells as 'E'\n        for row in range(len(board)):\n            if board[row][0] == \"O\":\n                dfs(row, 0)\n            if board[row][len(board[0]) - 1] == \"O\":\n                dfs(row, len(board[0]) - 1)\n\n        for col in range(len(board[0])):\n            if board[0][col] == \"O\":\n                dfs(0, col)\n            if board[len(board) - 1][col] == \"O\":\n                dfs(len(board) - 1, col)\n\n        # Mark internal 'O' cells as 'X' and restore 'E' cells to 'O'\n        for row in range(len(board)):\n            for col in range(len(board[0])):\n                if board[row][col] == \"O\":\n                    board[row][col] = \"X\"\n                elif board[row][col] == \"E\":\n                    board[row][col] = \"O\""
          }
        },
        {
          "id": 994,
          "title": "Rotting Oranges",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/rotting-oranges/",
          "description": "You are given an m x n grid of oranges where oranges[i][j] can be: 0: empty cell, 1: fresh orange, 2: rotten orange. Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten. Return the minimum number of minutes that must elapse until no fresh orange remains. If this is impossible, return -1.",
          "details": {
            "key_idea": "The problem is to determine the minimum time needed for all oranges to become rotten, considering that rotten oranges can also infect adjacent fresh oranges in each minute. We can model this problem using Breadth-First Search (BFS), where each minute corresponds to a level of the BFS traversal.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "from collections import deque\n\n\nclass Solution:\n    def orangesRotting(self, grid: List[List[int]]) -> int:\n        if not grid:\n            return -1\n\n        rows, cols = len(grid), len(grid[0])\n        fresh_count = 0  # Count of fresh oranges\n        rotten = deque()  # Queue to store coordinates of rotten oranges\n        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible adjacent cells\n\n        # Initialize the queue with coordinates of rotten oranges\n        for row in range(rows):\n            for col in range(cols):\n                if grid[row][col] == 2:\n                    rotten.append((row, col))\n                elif grid[row][col] == 1:\n                    fresh_count += 1\n\n        minutes = 0  # Timer\n\n        while rotten:\n            level_size = len(rotten)\n\n            for _ in range(level_size):\n                row, col = rotten.popleft()\n\n                for dr, dc in directions:\n                    new_row, new_col = row + dr, col + dc\n\n                    # Check if the new cell is within bounds and has a fresh orange\n                    if (\n                        0 <= new_row < rows\n                        and 0 <= new_col < cols\n                        and grid[new_row][new_col] == 1\n                    ):\n                        grid[new_row][new_col] = 2  # Infect the fresh orange\n                        fresh_count -= 1\n                        rotten.append((new_row, new_col))\n\n            if rotten:\n                minutes += 1\n\n        # If there are fresh oranges left, return -1; otherwise, return the elapsed minutes\n        return minutes if fresh_count == 0 else -1"
          }
        },
        {
          "id": 207,
          "title": "Course Schedule",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/course-schedule/",
          "description": "There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. Return true if you can finish all courses. Otherwise, return false.",
          "details": {
            "key_idea": "The problem can be reduced to detecting cycles in a directed graph. We can represent the course prerequisites as directed edges between nodes (courses). If there is a cycle in the graph, it means we can't complete all courses.",
            "time_complexity": "O(numCourses + len(prerequisites))",
            "space_complexity": "O(numCourses + len(prerequisites))",
            "python_solution": "class Solution:\n    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:\n        graph = {i: [] for i in range(numCourses)}\n        in_degree = [0] * numCourses\n\n        # Construct the graph and count in-degrees\n        for course, prereq in prerequisites:\n            graph[prereq].append(course)\n            in_degree[course] += 1\n\n        # Initialize a queue with nodes having in-degree zero\n        queue = collections.deque(\n            [course for course, degree in enumerate(in_degree) if degree == 0]\n        )\n\n        # Perform topological sorting and update in-degrees\n        while queue:\n            node = queue.popleft()\n            for neighbor in graph[node]:\n                in_degree[neighbor] -= 1\n                if in_degree[neighbor] == 0:\n                    queue.append(neighbor)\n\n        # If any course has in-degree greater than zero, there's a cycle\n        return all(degree == 0 for degree in in_degree)"
          }
        },
        {
          "id": 210,
          "title": "Course Schedule II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/course-schedule-ii/",
          "description": "There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. Return the ordering of courses you should take to finish all courses. If there are many valid orderings, return any of them. If it is impossible to finish all courses, return an empty array.",
          "details": {
            "key_idea": "This problem is an extension of the previous Course Schedule problem (LeetCode 207). We need to return the order in which courses can be taken. We can use the topological sorting approach to solve this.",
            "time_complexity": "O(numCourses + len(prerequisites))",
            "space_complexity": "O(numCourses + len(prerequisites))",
            "python_solution": "class Solution:\n    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:\n        graph = {i: [] for i in range(numCourses)}\n        in_degree = [0] * numCourses\n        order = []\n\n        # Construct the graph and count in-degrees\n        for course, prereq in prerequisites:\n            graph[prereq].append(course)\n            in_degree[course] += 1\n\n        # Initialize a queue with nodes having in-degree zero\n        queue = collections.deque(\n            [course for course, degree in enumerate(in_degree) if degree == 0]\n        )\n\n        # Perform topological sorting and update in-degrees\n        while queue:\n            node = queue.popleft()\n            order.append(node)\n            for neighbor in graph[node]:\n                in_degree[neighbor] -= 1\n                if in_degree[neighbor] == 0:\n                    queue.append(neighbor)\n\n        # If the order doesn't contain all courses, there's a cycle\n        return order if len(order) == numCourses else []"
          }
        },
        {
          "id": 684,
          "title": "Redundant Connection",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/redundant-connection/",
          "description": "In this problem, a tree is an undirected graph that is connected and has no cycles. You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge causes the graph to have a cycle. Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.",
          "details": {
            "key_idea": "This problem can be solved using the Union-Find (Disjoint Set Union) algorithm. We initialize each node as its own parent and iterate through the given edges. For each edge, we check if the nodes have the same parent. If they do, that means adding this edge will create a cycle, and it's the redundant edge. If they don't have the same parent, we merge their sets by updating one's parent to be the other.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:\n        n = len(edges)\n        parent = list(range(n + 1))  # Initialize each node as its own parent\n\n        def find(x):\n            if parent[x] != x:\n                parent[x] = find(parent[x])  # Path compression\n            return parent[x]\n\n        def union(x, y):\n            parent[find(x)] = find(y)\n\n        for edge in edges:\n            u, v = edge\n            if find(u) == find(v):\n                return edge\n            union(u, v)\n\n        return []"
          }
        },
        {
          "id": 323,
          "title": "Number of Connected Components in An Undirected Graph",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/",
          "description": "You are given an integer n which represents the number of nodes in the graph. You are also given a 2D integer array edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi. Return the number of connected components in the graph.",
          "details": {
            "key_idea": "This problem can be solved using Depth-First Search (DFS) or Breadth-First Search (BFS). We represent the given edges as an adjacency list, where each node points to its neighboring nodes. We then iterate through all nodes and perform a DFS/BFS from each unvisited node to explore all connected components. The number of times we need to start a new DFS/BFS corresponds to the number of connected components in the graph.",
            "time_complexity": "O(n + m)",
            "space_complexity": "O(n + m)",
            "python_solution": "from collections import defaultdict, deque\n\n\nclass Solution:\n    def countComponents(self, n: int, edges: List[List[int]]) -> int:\n        graph = defaultdict(list)\n        for u, v in edges:\n            graph[u].append(v)\n            graph[v].append(u)\n\n        def dfs(node):\n            visited.add(node)\n            for neighbor in graph[node]:\n                if neighbor not in visited:\n                    dfs(neighbor)\n\n        visited = set()\n        components = 0\n\n        for node in range(n):\n            if node not in visited:\n                components += 1\n                dfs(node)\n\n        return components"
          }
        },
        {
          "id": 261,
          "title": "Graph Valid Tree",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/graph-valid-tree/",
          "description": "Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.",
          "details": {
            "key_idea": "This problem can be solved using Depth-First Search (DFS) or Union-Find algorithm. We represent the given edges as an adjacency list, where each node points to its neighboring nodes. To determine whether the graph is a valid tree, we need to check two conditions: 1. The graph must be connected, i.e., there is a path between every pair of nodes. 2. There should be no cycles in the graph.",
            "time_complexity": "O(n + m)",
            "space_complexity": "O(n + m)",
            "python_solution": "from collections import defaultdict, deque\n\n\nclass Solution:\n    def validTree(self, n: int, edges: List[List[int]]) -> bool:\n        if len(edges) != n - 1:\n            return False\n\n        graph = defaultdict(list)\n        for u, v in edges:\n            graph[u].append(v)\n            graph[v].append(u)\n\n        visited = set()\n\n        def dfs(node, parent):\n            visited.add(node)\n            for neighbor in graph[node]:\n                if neighbor != parent:\n                    if neighbor in visited or not dfs(neighbor, node):\n                        return False\n            return True\n\n        # Check if the graph is connected\n        if not dfs(0, -1):\n            return False\n\n        return len(visited) == n"
          }
        },
        {
          "id": 127,
          "title": "Word Ladder",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/word-ladder/",
          "description": "A transformation sequence from wordBegin to wordEnd using a dictionary wordList is a sequence of wordsbeginWord -> s1 -> s2 -> ... -> sk such that: Every adjacent pair of words in the sequence form a transformation. That is, only one letter can be changed at a time between two adjacent words. Every word in the sequence including beginWord and endWord is in wordList. Return the shortest transformation sequence's length in words, or 0 if no such sequence exists.",
          "details": {
            "key_idea": "This problem can be solved using a breadth-first search (BFS) approach. We start with the given beginWord and perform a BFS to explore all possible word transformations, one character change at a time. We maintain a queue to track the current word and its transformation path. For each word in the queue, we generate all possible words by changing one character at a time and check if it's in the word list. If it is, we add it to the queue and mark it as visited. We continue this process until we reach the endWord or the queue is empty.",
            "time_complexity": "O(n * m)",
            "space_complexity": "O(n)",
            "python_solution": "from collections import deque\n\n\nclass Solution:\n    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:\n        wordSet = set(wordList)\n        if endWord not in wordSet:\n            return 0\n\n        queue = deque([(beginWord, 1)])  # Start from the beginWord with level 1\n        visited = set()\n\n        while queue:\n            word, level = queue.popleft()\n            if word == endWord:\n                return level\n\n            for i in range(len(word)):\n                for c in \"abcdefghijklmnopqrstuvwxyz\":\n                    new_word = word[:i] + c + word[i + 1 :]\n                    if new_word in wordSet and new_word not in visited:\n                        visited.add(new_word)\n                        queue.append((new_word, level + 1))\n\n        return 0"
          }
        }
      ]
    },
    {
      "topic_name": "Advanced Graphs",
      "problems": [
        {
          "id": 332,
          "title": "Reconstruct Itinerary",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/reconstruct-itinerary/",
          "description": "You are given a list of airline tickets where tickets[i] = [from_i, to_i] represent the flight leaving from airport from_i and arriving at airport to_i. Reconstruct the itinerary in order. All of the tickets belong to a man who departs from \"JFK\", thus the itinerary must begin with \"JFK\". If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary [\"JFK\", \"LGA\"] has a smaller lexical order than [\"JFK\",\"LGB\"]. All airports words are given in lowercase letters.",
          "details": {
            "key_idea": "The problem can be approached using a depth-first search (DFS) approach. We start from the \"JFK\" airport and explore all possible routes by visiting each airport exactly once. We use a dictionary to store the destinations for each source airport, and for each source airport, we sort the destinations in lexicographical order. This ensures that we visit the airports in the desired order.",
            "time_complexity": "O(n * log n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def findItinerary(self, tickets: List[List[str]]) -> List[str]:\n        graph = collections.defaultdict(list)\n\n        for start, end in sorted(tickets, reverse=True):\n            graph[start].append(end)\n\n        route = []\n\n        def dfs(node):\n            while graph[node]:\n                dfs(graph[node].pop())\n            route.append(node)\n\n        dfs(\"JFK\")\n\n        return route[::-1]"
          }
        },
        {
          "id": 1584,
          "title": "Min Cost to Connect All Points",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/min-cost-to-connect-all-points/",
          "description": "You are given an array of points where points[i] = [xi, yi] represents the coordinates of the ith point. The distance between two points is the Manhattan distance between them. Return the minimum cost to make all points connected. All points will be connected in the costliest manner.",
          "details": {
            "key_idea": "The problem can be solved using Kruskal's algorithm for finding the Minimum Spanning Tree (MST) of a graph. We start by calculating the distances between all pairs of points and then sort these distances along with their corresponding pairs. We initialize an empty MST and iterate through the sorted distances. For each distance, we check if the two points belong to different connected components in the MST using Union-Find. If they do, we add the distance to the MST and merge the components. We continue this process until all points are connected.",
            "time_complexity": "O(n^2 * log n)",
            "space_complexity": "O(n^2)",
            "python_solution": "class Solution:\n    def minCostConnectPoints(self, points: List[List[int]]) -> int:\n        def distance(p1, p2):\n            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])\n\n        n = len(points)\n        distances = []\n\n        for i in range(n):\n            for j in range(i + 1, n):\n                distances.append((distance(points[i], points[j]), i, j))\n\n        distances.sort()\n        parent = list(range(n))\n        rank = [0] * n\n        mst_cost = 0\n\n        def find(node):\n            if parent[node] != node:\n                parent[node] = find(parent[node])\n            return parent[node]\n\n        def union(node1, node2):\n            root1 = find(node1)\n            root2 = find(node2)\n            if root1 != root2:\n                if rank[root1] > rank[root2]:\n                    parent[root2] = root1\n                else:\n                    parent[root1] = root2\n                    if rank[root1] == rank[root2]:\n                        rank[root2] += 1\n\n        for distance, u, v in distances:\n            if find(u) != find(v):\n                union(u, v)\n                mst_cost += distance\n\n        return mst_cost"
          }
        },
        {
          "id": 743,
          "title": "Network Delay Time",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/network-delay-time/",
          "description": "You are given a network represented by a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target. We will send a signal from a given node k. Return the minimum time it takes for all n nodes to receive the signal. If it is impossible for all n nodes to receive the signal, return -1.",
          "details": {
            "key_idea": "The problem can be solved using Dijkstra's algorithm to find the shortest paths from a source node to all other nodes in a weighted graph. We start by creating an adjacency list representation of the graph. We maintain a priority queue to select the next node to visit based on the minimum distance. We initialize distances to all nodes as infinity except for the source node, which is set to 0. We continue exploring nodes and updating distances until the priority queue is empty. The maximum distance among all nodes will be the answer.",
            "time_complexity": "O(n * log(n) + m)",
            "space_complexity": "O(n + m)",
            "python_solution": "import heapq\nfrom collections import defaultdict\n\n\nclass Solution:\n    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:\n        # Create an adjacency list representation of the graph\n        graph = defaultdict(list)\n        for u, v, w in times:\n            graph[u].append((v, w))\n\n        # Initialize distances to all nodes as infinity except for the source node\n        distances = [float(\"inf\")] * (n + 1)\n        distances[k] = 0\n\n        # Priority queue to select the next node to visit based on the minimum distance\n        pq = [(0, k)]\n\n        while pq:\n            distance, node = heapq.heappop(pq)\n            if distance > distances[node]:\n                continue\n            for neighbor, weight in graph[node]:\n                if distance + weight < distances[neighbor]:\n                    distances[neighbor] = distance + weight\n                    heapq.heappush(pq, (distances[neighbor], neighbor))\n\n        # Find the maximum distance among all nodes\n        max_distance = max(distances[1:])\n\n        return max_distance if max_distance < float(\"inf\") else -1"
          }
        },
        {
          "id": 778,
          "title": "Swim in Rising Water",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/swim-in-rising-water/",
          "description": "You are given an n x n integer matrix grid where each value grid[i][j] represents the elevation at that point (i,j). The rain source and rain \nstart to pour from the top-left cell. You need to find the minimum time t that you can reach the bottom-right cell (n - 1, n - 1) starting from the top-left cell (0, 0). You can move 4-directionally (up, down, left, or right) from a cell to another cell.",
          "details": {
            "key_idea": "The problem can be approached using a binary search along with depth-first search (DFS). We can perform a binary search on the range of possible time values, from the minimum value (0) to the maximum value (N * N). For each time value, we perform a DFS to check if it's possible to reach the bottom-right cell from the top-left cell without encountering cells with heights greater than the current time value. If a valid path exists, we narrow down the search to the left half of the time range; otherwise, we search in the right half.",
            "time_complexity": "O(N^2 * log(N^2))",
            "space_complexity": "O(N^2)",
            "python_solution": "class Solution:\n    def swimInWater(self, grid: List[List[int]]) -> int:\n        def dfs(i, j, visited, time):\n            if i < 0 or i >= N or j < 0 or j >= N or visited[i][j] or grid[i][j] > time:\n                return False\n            if i == N - 1 and j == N - 1:\n                return True\n            visited[i][j] = True\n            return (\n                dfs(i + 1, j, visited, time)\n                or dfs(i - 1, j, visited, time)\n                or dfs(i, j + 1, visited, time)\n                or dfs(i, j - 1, visited, time)\n            )\n\n        N = len(grid)\n        left, right = 0, N * N\n\n        while left < right:\n            mid = (left + right) // 2\n            visited = [[False] * N for _ in range(N)]\n            if dfs(0, 0, visited, mid):\n                right = mid\n            else:\n                left = mid + 1\n\n        return left"
          }
        },
        {
          "id": 269,
          "title": "Alien Dictionary",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/alien-dictionary/",
          "description": "There is a new alien language which uses the English letter. The following is a list of strings sorted lexicographically by the rules of this new language. Derive out the order of letters in this language. You may assume that for each pair of adjacent strings in the list, there is at least one difference in characters between them. This difference is the first difference found from the left, i.e. if one string is a prefix of another, the shorter string is the first lexicographically. If the order is invalid, return \"\".",
          "details": {
            "key_idea": "The problem can be solved using topological sorting. The given words can be thought of as directed edges between characters of adjacent words. We can build a graph where each character is a node, and the edges represent the order between characters. Then, we can perform topological sorting to find the correct order of characters.",
            "time_complexity": "O(N + E + V)",
            "space_complexity": "O(V + E)",
            "python_solution": "from collections import defaultdict, deque\n\n\nclass Solution:\n    def alienOrder(self, words: List[str]) -> str:\n        graph = defaultdict(list)\n        in_degree = defaultdict(int)\n\n        for i in range(len(words) - 1):\n            word1, word2 = words[i], words[i + 1]\n            for j in range(min(len(word1), len(word2))):\n                if word1[j] != word2[j]:\n                    graph[word1[j]].append(word2[j])\n                    in_degree[word2[j]] += 1\n                    break\n\n        queue = deque(char for char, indeg in in_degree.items() if indeg == 0)\n        result = []\n\n        while queue:\n            char = queue.popleft()\n            result.append(char)\n            for neighbor in graph[char]:\n                in_degree[neighbor] -= 1\n                if in_degree[neighbor] == 0:\n                    queue.append(neighbor)\n\n        if len(result) < len(in_degree):\n            return \"\"\n        return \"\".join(result)"
          }
        },
        {
          "id": 787,
          "title": "Cheapest Flights Within K Stops",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/cheapest-flights-within-k-stops/",
          "description": "There are n cities connected by m flights, where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei. You are also given an integer src, the source city, and an integer dst, the destination city, and an integer k, the maximum number of stops that you can make in your journey.",
          "details": {
            "key_idea": "The problem can be solved using Dijkstra's algorithm with a modified priority queue (min-heap) and BFS (Breadth-First Search) approach. We can create a graph where each node represents a city, and the edges represent flights between cities with associated costs. We use a priority queue to explore the nodes in a way that prioritizes the minimum cost. We continue the BFS process until we reach the destination city or exhaust the maximum number of allowed stops (K+1).",
            "time_complexity": "O(E * log(V))",
            "space_complexity": "O(V + E)",
            "python_solution": "import heapq\nimport math\nfrom typing import List\n\n\nclass Solution:\n    def findCheapestPrice(\n        self, n: int, flights: List[List[int]], src: int, dst: int, max_stops: int\n    ) -> int:\n        graph = [[] for _ in range(n)]\n        min_heap = [\n            (0, src, max_stops + 1)  # (total_cost, current_city, remaining_stops)\n        ]\n        distances = [[math.inf] * (max_stops + 2) for _ in range(n)]\n\n        for u, v, w in flights:\n            graph[u].append((v, w))\n\n        while min_heap:\n            total_cost, current_city, remaining_stops = heapq.heappop(min_heap)\n            if current_city == dst:\n                return total_cost\n            if remaining_stops > 0:\n                for neighbor, cost in graph[current_city]:\n                    new_cost = total_cost + cost\n                    if new_cost < distances[neighbor][remaining_stops - 1]:\n                        distances[neighbor][remaining_stops - 1] = new_cost\n                        heapq.heappush(\n                            min_heap, (new_cost, neighbor, remaining_stops - 1)\n                        )\n\n        return -1"
          }
        }
      ]
    },
    {
      "topic_name": "1-D Dynamic Programming",
      "problems": [
        {
          "id": 70,
          "title": "Climbing Stairs",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/climbing-stairs/",
          "description": "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
          "details": {
            "key_idea": "To climb to the n-th stair, you have two options, 1. You can climb from the (n-1)-th stair. 2. You can climb from the (n-2)-th stair. So, the number of ways to reach the n-th stair is the sum of the number of ways to reach the (n-1)-th and (n-2)-th stairs. This forms a Fibonacci sequence where f(n) = f(n-1) + f(n-2).",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def climbStairs(self, n: int) -> int:\n        if n <= 2:\n            return n\n\n        prev1 = 1  # Number of ways to reach the 1st stair\n        prev2 = 2  # Number of ways to reach the 2nd stair\n\n        for i in range(3, n + 1):\n            current = prev1 + prev2\n            prev1, prev2 = prev2, current  # Update for the next iteration\n\n        return prev2  # Number of ways to reach the n-th stair"
          }
        },
        {
          "id": 746,
          "title": "Min Cost Climbing Stairs",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/min-cost-climbing-stairs/",
          "description": "You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps. You can either start from the step with cost 0, or the step with cost 1. Return the minimum cost to reach the top of the floor.",
          "details": {
            "key_idea": "To find the minimum cost to reach the top of the stairs, we can use dynamic programming. We start by creating a list dp where dp[i] represents the minimum cost to reach the i-th stair. We initialize dp[0] and dp[1] to the costs of the 0-th and 1-st stairs, as there's no cost to reach them. Then, we iterate from the 2nd stair to the top, and for each stair i, we calculate dp[i] as the minimum of either reaching it from dp[i-1] (one step) or dp[i-2] (two steps), plus the cost of the current stair. The minimum cost to reach the top will be either dp[n-1] or dp[n-2], where n is the number of stairs, as we can reach the top by either taking one step from the last stair or two steps from the second-to-last stair.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def minCostClimbingStairs(self, cost: List[int]) -> int:\n        n = len(cost)\n        if n <= 1:\n            return 0  # No cost if there are 0 or 1 stairs\n\n        dp = [0] * n  # Initialize a list to store minimum costs\n\n        # Base cases\n        dp[0] = cost[0]\n        dp[1] = cost[1]\n\n        for i in range(2, n):\n            dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]\n\n        return min(dp[n - 1], dp[n - 2])"
          }
        },
        {
          "id": 198,
          "title": "House Robber",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/house-robber/",
          "description": "You are robbing houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night. Given the integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.",
          "details": {
            "key_idea": "The key idea to solve this problem is to use dynamic programming. We create a list `dp` where `dp[i]` represents the maximum amount of money that can be robbed up to the `i-th` house. To populate `dp`, we iterate through the houses from left to right. For each house `i`, we have two choices: 1. Rob the current house, which means we add the money from the current house (`nums[i]`) to the amount robbed from two houses before (`dp[i-2]`). 2. Skip the current house and take the maximum amount robbed from the previous house (`dp[i-1]`). We choose the maximum of these two options and update `dp[i]` accordingly. Finally, `dp[-1]` will contain the maximum amount that can be robbed.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def rob(self, nums: List[int]) -> int:\n        n = len(nums)\n        if n == 0:\n            return 0\n        if n == 1:\n            return nums[0]\n\n        dp = [0] * n\n        dp[0] = nums[0]\n        dp[1] = max(nums[0], nums[1])\n\n        for i in range(2, n):\n            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])\n\n        return dp[-1]"
          }
        },
        {
          "id": 213,
          "title": "House Robber II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/house-robber-ii/",
          "description": "Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police. Because all houses are arranged in a circle, the first house and the last house are mutually exclusive.",
          "details": {
            "key_idea": "The key idea for solving this problem is to recognize that it's an extension of the House Robber I problem (LeetCode 198). Since we cannot rob adjacent houses, we have two scenarios: 1. Rob the first house and exclude the last house. 2. Exclude the first house and consider robbing the last house. We calculate the maximum amount for both scenarios using the House Robber I algorithm and return the maximum of the two results.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def rob(self, nums: List[int]) -> int:\n        def houseRobber(nums: List[int]) -> int:\n            n = len(nums)\n            if n == 0:\n                return 0\n            if n == 1:\n                return nums[0]\n\n            prev = 0\n            curr = 0\n\n            for num in nums:\n                temp = curr\n                curr = max(prev + num, curr)\n                prev = temp\n\n            return curr\n\n        if len(nums) == 1:\n            return nums[0]\n\n        # Rob first house and exclude the last house, or exclude the first house and rob the last house.\n        return max(houseRobber(nums[1:]), houseRobber(nums[:-1]))"
          }
        },
        {
          "id": 5,
          "title": "Longest Palindromic Substring",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/longest-palindromic-substring/",
          "description": "Given a string s, return the longest palindromic substring in s.",
          "details": {
            "key_idea": "The key idea for solving this problem is to expand around each character in the string and check for palindromes. We will consider each character as the center of a potential palindrome and expand outwards while checking if the characters at both ends are equal. We need to handle two cases: palindromes with odd length (centered at a single character) and palindromes with even length (centered between two characters).",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def longestPalindrome(self, s: str) -> str:\n        def expandAroundCenter(left: int, right: int) -> str:\n            # Expand around the center while the characters at both ends are equal.\n            while left >= 0 and right < len(s) and s[left] == s[right]:\n                left -= 1\n                right += 1\n            # Return the palindrome substring.\n            return s[left + 1 : right]\n\n        if len(s) < 2:\n            return s\n\n        longest = \"\"\n\n        for i in range(len(s)):\n            # Check for odd-length palindromes.\n            palindrome1 = expandAroundCenter(i, i)\n            if len(palindrome1) > len(longest):\n                longest = palindrome1\n\n            # Check for even-length palindromes.\n            palindrome2 = expandAroundCenter(i, i + 1)\n            if len(palindrome2) > len(longest):\n                longest = palindrome2\n\n        return longest"
          }
        },
        {
          "id": 647,
          "title": "Palindromic Substrings",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/palindromic-substrings/",
          "description": "Given a string s, return the number of palindromic substrings in it.",
          "details": {
            "key_idea": "The key idea for solving this problem is to expand around each character in the string and count the palindromic substrings. We will consider each character as the center of a potential palindrome and expand outwards while checking if the characters at both ends are equal. We need to handle two cases: palindromes with odd length (centered at a single character) and palindromes with even length (centered between two characters).",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def countSubstrings(self, s: str) -> int:\n        def expandAroundCenter(left: int, right: int) -> int:\n            count = 0\n            # Expand around the center while the characters at both ends are equal.\n            while left >= 0 and right < len(s) and s[left] == s[right]:\n                left -= 1\n                right += 1\n                count += 1\n            return count\n\n        if not s:\n            return 0\n\n        total_palindromes = 0\n\n        for i in range(len(s)):\n            # Check for odd-length palindromes.\n            total_palindromes += expandAroundCenter(i, i)\n\n            # Check for even-length palindromes.\n            total_palindromes += expandAroundCenter(i, i + 1)\n\n        return total_palindromes"
          }
        },
        {
          "id": 91,
          "title": "Decode Ways",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/decode-ways/",
          "description": "A message containing letters from A-Z can be encoded into numbers using the following mapping: 'A' -> \"1\", 'B' -> \"2\", ..., 'Z' -> \"26\". Given a string s containing only digits, return the total number of ways to decode it.",
          "details": {
            "key_idea": "The key idea for solving this problem is to use dynamic programming to count the number of ways to decode the given string. We will create a DP array where dp[i] represents the number of ways to decode the string s[:i] (the first i characters of the string).",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def numDecodings(self, s: str) -> int:\n        n = len(s)\n\n        # Initialize a DP array to store the number of ways to decode substrings.\n        dp = [0] * (n + 1)\n\n        # Base cases:\n        dp[0] = 1  # An empty string can be decoded in one way.\n        dp[1] = (\n            1 if s[0] != \"0\" else 0\n        )  # The first character can be decoded in one way if it's not '0'.\n\n        # Fill in the DP array.\n        for i in range(2, n + 1):\n            # Check the one-digit and two-digit possibilities.\n            one_digit = int(s[i - 1])\n            two_digits = int(s[i - 2 : i])\n\n            # If the one-digit is not '0', it can be decoded in the same way as dp[i-1].\n            if one_digit != 0:\n                dp[i] += dp[i - 1]\n\n            # If the two-digit is between 10 and 26, it can be decoded in the same way as dp[i-2].\n            if 10 <= two_digits <= 26:\n                dp[i] += dp[i - 2]\n\n        return dp[n]"
          }
        },
        {
          "id": 322,
          "title": "Coin Change",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/coin-change/",
          "description": "You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to find the minimum number of coins needed to make up the given amount. We will create a DP array where dp[i] represents the minimum number of coins needed to make up the amount i.",
            "time_complexity": "O(amount * n)",
            "space_complexity": "O(amount)",
            "python_solution": "class Solution:\n    def coinChange(self, coins: List[int], amount: int) -> int:\n        # Initialize the DP array with a maximum value to represent impossible cases.\n        dp = [float(\"inf\")] * (amount + 1)\n\n        # Base case: It takes 0 coins to make up the amount of 0.\n        dp[0] = 0\n\n        # Iterate through the DP array and update the minimum number of coins needed.\n        for coin in coins:\n            for i in range(coin, amount + 1):\n                dp[i] = min(dp[i], dp[i - coin] + 1)\n\n        # If dp[amount] is still float('inf'), it means it's impossible to make up the amount.\n        # Otherwise, dp[amount] contains the minimum number of coins needed.\n        return dp[amount] if dp[amount] != float(\"inf\") else -1"
          }
        },
        {
          "id": 152,
          "title": "Maximum Product Subarray",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/maximum-product-subarray/",
          "description": "Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to keep track of the maximum and minimum product ending at each position in the array. Since multiplying a negative number by a negative number results in a positive number, we need to keep track of both the maximum and minimum products to handle negative numbers.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def maxProduct(self, nums: List[int]) -> int:\n        # Initialize variables to keep track of the maximum and minimum product ending at the current position.\n        max_product = min_product = result = nums[0]\n\n        # Iterate through the array starting from the second element.\n        for i in range(1, len(nums)):\n            # If the current element is negative, swap max_product and min_product\n            # because multiplying a negative number can turn the maximum into the minimum.\n            if nums[i] < 0:\n                max_product, min_product = min_product, max_product\n\n            # Update max_product and min_product based on the current element.\n            max_product = max(nums[i], max_product * nums[i])\n            min_product = min(nums[i], min_product * nums[i])\n\n            # Update the overall result with the maximum product found so far.\n            result = max(result, max_product)\n\n        return result"
          }
        },
        {
          "id": 139,
          "title": "Word Break",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/word-break/",
          "description": "Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to determine if it's possible to break the input string into words from the wordDict. We can create a boolean array dp, where dp[i] is True if we can break the substring s[0:i] into words from the wordDict.",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def wordBreak(self, s: str, wordDict: List[str]) -> bool:\n        # Create a set for faster word lookup.\n        wordSet = set(wordDict)\n\n        # Initialize a boolean array dp where dp[i] is True if s[0:i] can be broken into words.\n        dp = [False] * (len(s) + 1)\n        dp[0] = True  # An empty string can always be broken.\n\n        # Iterate through the string.\n        for i in range(1, len(s) + 1):\n            for j in range(i):\n                # Check if the substring s[j:i] is in the wordDict and if s[0:j] can be broken.\n                if dp[j] and s[j:i] in wordSet:\n                    dp[i] = True\n                    break  # No need to continue checking.\n\n        return dp[len(s)]"
          }
        },
        {
          "id": 300,
          "title": "Longest Increasing Subsequence",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/longest-increasing-subsequence/",
          "description": "Given an integer array nums sorted in ascending order, return the length of the longest consecutive elements sequence. You must write an algorithm that runs in O(n) time.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to find the length of the longest increasing subsequence. We create a dynamic programming array dp, where dp[i] represents the length of the longest increasing subsequence ending at index i.",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def lengthOfLIS(self, nums: List[int]) -> int:\n        if not nums:\n            return 0\n\n        # Initialize a dynamic programming array dp with all values set to 1.\n        dp = [1] * len(nums)\n\n        # Iterate through the array to find the longest increasing subsequence.\n        for i in range(len(nums)):\n            for j in range(i):\n                if nums[i] > nums[j]:\n                    dp[i] = max(dp[i], dp[j] + 1)\n\n        # Return the maximum value in dp, which represents the length of the longest increasing subsequence.\n        return max(dp)"
          }
        },
        {
          "id": 416,
          "title": "Partition Equal Subset Sum",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/partition-equal-subset-sum/",
          "description": "Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to determine if it's possible to partition the input array into two subsets with equal sums. We create a dynamic programming array dp, where dp[i][j] represents whether it's possible to form a subset with a sum of 'j' using the first 'i' elements of the input array.",
            "time_complexity": "O(n * sum(nums))",
            "space_complexity": "O(n * sum(nums))",
            "python_solution": "class Solution:\n    def canPartition(self, nums: List[int]) -> bool:\n        total_sum = sum(nums)\n\n        # If the total sum is odd, it's impossible to partition into two equal subsets.\n        if total_sum % 2 != 0:\n            return False\n\n        target_sum = total_sum // 2\n\n        # Initialize a dynamic programming array dp with all values set to False.\n        dp = [False] * (target_sum + 1)\n\n        # It's always possible to achieve a sum of 0 using an empty subset.\n        dp[0] = True\n\n        for num in nums:\n            for i in range(target_sum, num - 1, -1):\n                # If it's possible to achieve a sum of 'i - num' using a subset,\n                # then it's also possible to achieve a sum of 'i' using a subset.\n                dp[i] = dp[i] or dp[i - num]\n\n        return dp[target_sum]"
          }
        }
      ]
    },
    {
      "topic_name": "2-D Dynamic Programing",
      "problems": [
        {
          "id": 62,
          "title": "Unique Paths",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/unique-paths/",
          "description": "A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below). The robot can only either try to move down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below). How many possible unique paths are there?",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to count the number of unique paths from the top-left corner to the bottom-right corner of a grid. We create a dynamic programming grid 'dp', where dp[i][j] represents the number of unique paths to reach cell (i, j) from the top-left corner.",
            "time_complexity": "O(m*n)",
            "space_complexity": "O(m*n)",
            "python_solution": "class Solution:\n    def uniquePaths(self, m: int, n: int) -> int:\n        # Initialize a 2D dp grid of size m x n with all values set to 1.\n        dp = [[1] * n for _ in range(m)]\n\n        # Fill in the dp grid using dynamic programming.\n        for i in range(1, m):\n            for j in range(1, n):\n                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]\n\n        # The value in dp[m-1][n-1] represents the number of unique paths.\n        return dp[m - 1][n - 1]"
          }
        },
        {
          "id": 1143,
          "title": "Longest Common Subsequence",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/longest-common-subsequence/",
          "description": "Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to find the length of the longest common subsequence (LCS) between two strings. We create a 2D dp array where dp[i][j] represents the length of the LCS between the first i characters of text1 and the first j characters of text2.",
            "time_complexity": "O(m*n)",
            "space_complexity": "O(m*n)",
            "python_solution": "class Solution:\n    def longestCommonSubsequence(self, text1: str, text2: str) -> int:\n        m, n = len(text1), len(text2)\n\n        # Create a 2D dp array of size (m+1) x (n+1) and initialize it with zeros.\n        dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n        # Fill in the dp array using dynamic programming.\n        for i in range(1, m + 1):\n            for j in range(1, n + 1):\n                if text1[i - 1] == text2[j - 1]:\n                    dp[i][j] = dp[i - 1][j - 1] + 1\n                else:\n                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n\n        # The value in dp[m][n] represents the length of the LCS.\n        return dp[m][n]"
          }
        },
        {
          "id": 309,
          "title": "Best Time to Buy and Sell Stock with Cooldown",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/",
          "description": "You are given an array prices where prices[i] is the price of a given stock on the ith day. Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay a cooldown period of 1 day between two transactions. That is, you cannot buy on the next day after you have sold your stock.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to keep track of the maximum profit we can obtain at each day 'i', considering the actions we can take: buy, sell, or do nothing (cooldown).",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def maxProfit(self, prices: List[int]) -> int:\n        if not prices:\n            return 0\n\n        # Initialize variables to represent the maximum profit after each action.\n        buy = -prices[\n            0\n        ]  # Maximum profit after buying on day 0 (negative because we spend money).\n        sell = 0  # Maximum profit after selling on day 0 (no profit yet).\n        cooldown = 0  # Maximum profit after cooldown on day 0 (no profit yet).\n\n        for i in range(1, len(prices)):\n            # To maximize profit on day 'i', we can either:\n\n            # 1. Buy on day 'i'. We subtract the price of the stock from the maximum profit after cooldown on day 'i-2'.\n            new_buy = max(buy, cooldown - prices[i])\n\n            # 2. Sell on day 'i'. We add the price of the stock to the maximum profit after buying on day 'i-1'.\n            new_sell = buy + prices[i]\n\n            # 3. Do nothing (cooldown) on day 'i'. We take the maximum of the maximum profit after cooldown on day 'i-1' and after selling on day 'i-1'.\n            new_cooldown = max(cooldown, sell)\n\n            # Update the variables for the next iteration.\n            buy, sell, cooldown = new_buy, new_sell, new_cooldown\n\n        # The maximum profit will be the maximum of the profit after selling on the last day and the profit after cooldown on the last day.\n        return max(sell, cooldown)"
          }
        },
        {
          "id": 518,
          "title": "Coin Change II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/coin-change-ii/",
          "description": "You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to count the number of combinations that make up each amount from 0 to 'amount'.",
            "time_complexity": "O(amount * n)",
            "space_complexity": "O(amount)",
            "python_solution": "class Solution:\n    def change(self, amount: int, coins: List[int]) -> int:\n        # Initialize a 1D array dp to store the number of combinations for each amount from 0 to amount.\n        dp = [0] * (amount + 1)\n\n        # There is one way to make amount 0 (by not using any coins).\n        dp[0] = 1\n\n        # Iterate through each coin denomination.\n        for coin in coins:\n            # Update the dp array for each amount from coin to amount.\n            for i in range(coin, amount + 1):\n                dp[i] += dp[i - coin]\n\n        # The dp[amount] contains the number of combinations to make the target amount.\n        return dp[amount]"
          }
        },
        {
          "id": 494,
          "title": "Target Sum",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/target-sum/",
          "description": "You are given an integer array nums and an integer target. You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenating all the integers. For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 to make the expression \"+2-1\". Return the number of different expressions that you can build, which evaluates to target.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to count the number of ways to reach a certain sum by assigning either a positive or negative sign to each element in the input array.",
            "time_complexity": "O(n * sum)",
            "space_complexity": "O(sum)",
            "python_solution": "class Solution:\n    def findTargetSumWays(self, nums: List[int], S: int) -> int:\n        # Calculate the sum of all elements in the input array 'nums'.\n        total_sum = sum(nums)\n\n        # If the total sum is less than the target sum 'S', it's not possible to reach 'S'.\n        if (total_sum + S) % 2 != 0 or total_sum < abs(S):\n            return 0\n\n        # Calculate the target sum for positive signs. (total_sum + S) / 2\n        target = (total_sum + S) // 2\n\n        # Initialize a 1D array 'dp' to store the number of ways to reach each sum from 0 to 'target'.\n        dp = [0] * (target + 1)\n\n        # There is one way to reach a sum of 0 (by not selecting any element).\n        dp[0] = 1\n\n        # Iterate through each element in the input array 'nums'.\n        for num in nums:\n            # Update the 'dp' array for each sum from 'target' to 'num'.\n            for i in range(target, num - 1, -1):\n                dp[i] += dp[i - num]\n\n        # The 'dp[target]' contains the number of ways to reach the target sum 'S'.\n        return dp[target]"
          }
        },
        {
          "id": 329,
          "title": "Longest Increasing Path in a Matrix",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/longest-increasing-path-in-a-matrix/",
          "description": "Given an m x n integers matrix, return the length of the longest increasing path in matrix. From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to find the longest increasing path in a matrix. We can start from each cell and perform a depth-first search (DFS) to explore the neighboring cells with larger values.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:\n        if not matrix:\n            return 0\n\n        # Define directions for moving to neighboring cells: up, down, left, right.\n        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]\n\n        # Function to perform DFS from a given cell (i, j).\n        def dfs(i, j):\n            # If the result for this cell is already calculated, return it.\n            if dp[i][j] != -1:\n                return dp[i][j]\n\n            # Initialize the result for this cell to 1 (counting itself).\n            dp[i][j] = 1\n\n            # Explore the four neighboring cells.\n            for dx, dy in directions:\n                x, y = i + dx, j + dy\n                if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:\n                    # If the neighboring cell has a larger value, perform DFS.\n                    dp[i][j] = max(dp[i][j], 1 + dfs(x, y))\n\n            return dp[i][j]\n\n        m, n = len(matrix), len(matrix[0])\n        dp = [[-1] * n for _ in range(m)]  # Memoization table to store results.\n        max_path = 0\n\n        # Start DFS from each cell in the matrix.\n        for i in range(m):\n            for j in range(n):\n                max_path = max(max_path, dfs(i, j))\n\n        return max_path"
          }
        },
        {
          "id": 115,
          "title": "Distinct Subsequences",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/distinct-subsequences/",
          "description": "Given two strings s and t, return the number of distinct subsequences of s which equals t.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to count the number of distinct subsequences of the string 's' that match the string 't'. We can create a 1D DP array dp, where dp[j] represents the number of distinct subsequences of 's' that match 't' up to index 'j'. We can build the DP array based on the following rules: 1. Initialize dp[0] to 1 because there is one way to match an empty string '\"\"' to another empty string '\"\"'. 2. For each character in 's', update dp[j] based on the following conditions: - If s[i] == t[j], we can either match the current characters (s[i] and t[j]), which contributes dp[j-1] to the count, or we can skip the current character in 's', which contributes dp[j] to the count. - If s[i] != t[j], we can only skip the current character in 's', which contributes dp[j] to the count.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(n)",
            "python_solution": "# This solution only passes 65/66 testcases for some reason. Tried other solutions and they don't work either, so it's probably a faulty testcase.\n# If you have a solution that passes all testcases, please open a pr.\n\n\nclass Solution:\n    def numDistinct(self, s: str, t: str) -> int:\n        m, n = len(s), len(t)\n\n        # Create a 2D table dp to store the number of distinct subsequences.\n        dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n        # Initialize the first row of dp. There is one way to form an empty subsequence.\n        for i in range(m + 1):\n            dp[i][0] = 1\n\n        # Fill the dp table using dynamic programming.\n        for i in range(1, m + 1):\n            for j in range(1, n + 1):\n                # If the characters match, we have two options:\n                # 1. Include the current character in the subsequence (dp[i-1][j-1] ways).\n                # 2. Exclude the current character (dp[i-1][j] ways).\n                if s[i - 1] == t[j - 1]:\n                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]\n                else:\n                    # If the characters don't match, we can only exclude the current character.\n                    dp[i][j] = dp[i - 1][j]\n\n        return dp[m][n]"
          }
        },
        {
          "id": 72,
          "title": "Edit Distance",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/edit-distance/",
          "description": "Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to calculate the minimum number of operations (insert, delete, or replace) required to transform one string 'word1' into another string 'word2'. We can define a 2D table 'dp' where dp[i][j] represents the minimum edit distance between the first 'i' characters of 'word1' and the first 'j' characters of 'word2'.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def minDistance(self, word1: str, word2: str) -> int:\n        m, n = len(word1), len(word2)\n\n        # Create a 2D table dp to store the minimum edit distance.\n        dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n        # Initialize the first row and first column of dp.\n        for i in range(m + 1):\n            dp[i][0] = i\n        for j in range(n + 1):\n            dp[0][j] = j\n\n        for i in range(1, m + 1):\n            for j in range(1, n + 1):\n                # If the characters match, no additional operation is needed.\n                if word1[i - 1] == word2[j - 1]:\n                    dp[i][j] = dp[i - 1][j - 1]\n                else:\n                    # Choose the minimum of the three possible operations:\n                    # 1. Insert a character (dp[i][j - 1] + 1)\n                    # 2. Delete a character (dp[i - 1][j] + 1)\n                    # 3. Replace a character (dp[i - 1][j - 1] + 1)\n                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1\n\n        return dp[m][n]"
          }
        },
        {
          "id": 312,
          "title": "Burst Balloons",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/burst-balloons/",
          "description": "Given n balloons, indexed from 0 to n - 1. Each balloon has a number painted on it represented by the array nums. You are asked to burst all the balloons. If you burst the ith balloon, you will get nums[i] * left_bound * right_bound coins. Here left_bound and right_bound are adjacent indices of i, and if no adjacent indices exist, then the index is assumed to be 1. Find the maximum coins you can collect by bursting the coins.",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to calculate the maximum coins that can be obtained by bursting balloons in a certain order. We create a 2D table 'dp' where dp[i][j] represents the maximum coins obtained by bursting balloons from 'i' to 'j' (exclusive). We iterate through different balloon ranges and choose the best order to burst the balloons.",
            "time_complexity": "O(n^3)",
            "space_complexity": "O(n^2)",
            "python_solution": "class Solution:\n    def maxCoins(self, nums: List[int]) -> int:\n        n = len(nums)\n\n        # Add virtual balloons at the beginning and end with a value of 1.\n        nums = [1] + nums + [1]\n\n        # Create a 2D table dp to store the maximum coins.\n        dp = [[0] * (n + 2) for _ in range(n + 2)]\n\n        # Iterate through different balloon ranges.\n        for length in range(2, n + 2):\n            for left in range(0, n + 2 - length):\n                right = left + length\n                for k in range(left + 1, right):\n                    # Choose the best order to burst balloons in the range [left, right].\n                    dp[left][right] = max(\n                        dp[left][right],\n                        nums[left] * nums[k] * nums[right] + dp[left][k] + dp[k][right],\n                    )\n\n        return dp[0][n + 1]"
          }
        },
        {
          "id": 10,
          "title": "Regular Expression Matching",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/regular-expression-matching/",
          "description": "Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'. '.' matches any single character. '*' matches zero or more of the preceding element. The matching should cover the entire input string (not partial).",
          "details": {
            "key_idea": "The key idea is to use dynamic programming to solve this problem. We create a 2D table 'dp' where dp[i][j] represents whether the first 'i' characters in the string 's' match the first 'j' characters in the pattern 'p'. We fill this table based on the characters in 's' and 'p' and previous results.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m * n)",
            "python_solution": "class Solution:\n    def isMatch(self, s: str, p: str) -> bool:\n        m, n = len(s), len(p)\n\n        # Create a 2D table dp to store whether s[:i] matches p[:j].\n        dp = [[False] * (n + 1) for _ in range(m + 1)]\n\n        # Base case: empty string matches empty pattern.\n        dp[0][0] = True\n\n        # Fill the first row of dp based on '*' in the pattern.\n        for j in range(2, n + 1):\n            if p[j - 1] == \"*\":\n                dp[0][j] = dp[0][j - 2]\n\n        # Fill the dp table based on characters in s and p.\n        for i in range(1, m + 1):\n            for j in range(1, n + 1):\n                if p[j - 1] == s[i - 1] or p[j - 1] == \".\":\n                    dp[i][j] = dp[i - 1][j - 1]\n                elif p[j - 1] == \"*\":\n                    dp[i][j] = dp[i][j - 2] or (\n                        dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == \".\")\n                    )\n\n        return dp[m][n]"
          }
        }
      ]
    },
    {
      "topic_name": "Greedy",
      "problems": [
        {
          "id": 53,
          "title": "Maximum Subarray",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/maximum-subarray/",
          "description": "Given an integer array nums, find the contiguous non-empty subarray (containing at least one number) which has the largest sum and return its sum.",
          "details": {
            "key_idea": "We can solve this problem using the Kadane's algorithm. The idea is to iterate through the array and keep track of two variables: `max_sum` which stores the maximum sum ending at the current index, and `current_sum` which stores the maximum sum of subarray ending at the current index. For each element, we update `current_sum` by taking the maximum of the current element itself or adding it to the `current_sum` of the previous index. If `current_sum` becomes negative, we reset it to 0. Meanwhile, we update `max_sum` with the maximum of `max_sum` and `current_sum` at each step. After iterating through the entire array, `max_sum` will hold the maximum subarray sum.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def maxSubArray(self, nums: List[int]) -> int:\n        max_sum = float(\"-inf\")\n        current_sum = 0\n\n        for num in nums:\n            current_sum = max(num, current_sum + num)\n            max_sum = max(max_sum, current_sum)\n\n        return max_sum"
          }
        },
        {
          "id": 55,
          "title": "Jump Game",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/jump-game/",
          "description": "You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position. Return true if you can reach the last index, or false otherwise.",
          "details": {
            "key_idea": "We can solve this problem using a greedy approach. The idea is to keep track of the maximum reachable index as we iterate through the array. At each index, we update the maximum reachable index by taking the maximum of the current index plus the value at that index (which represents the maximum jump length) and the previous maximum reachable index. If the maximum reachable index surpasses or equals the last index, we know it's possible to reach the end.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def canJump(self, nums: List[int]) -> bool:\n        max_reachable = 0\n\n        for i, jump_length in enumerate(nums):\n            if i > max_reachable:\n                return False\n            max_reachable = max(max_reachable, i + jump_length)\n\n        return True"
          }
        },
        {
          "id": 45,
          "title": "Jump Game II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/jump-game-ii/",
          "description": "You are given a 0-indexed integer array nums of length n. You are initially positioned at nums[0]. Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where: 0 <= j <= nums[i] and i + j < n. Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can always reach nums[n - 1].",
          "details": {
            "key_idea": "We can solve this problem using a greedy approach. The idea is to keep track of the farthest reachable index and the current end of the interval. As we iterate through the array, we update the farthest reachable index based on the maximum jump length at the current index. When the current index reaches the end of the interval, we update the end of the interval to the farthest reachable index and increment the jump count. This ensures that we always make the jump with the maximum reach.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def jump(self, nums: List[int]) -> int:\n        jumps = 0\n        cur_end = 0\n        farthest_reachable = 0\n\n        for i in range(len(nums) - 1):\n            farthest_reachable = max(farthest_reachable, i + nums[i])\n            if i == cur_end:\n                jumps += 1\n                cur_end = farthest_reachable\n\n        return jumps"
          }
        },
        {
          "id": 134,
          "title": "Gas Station",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/gas-station/",
          "description": "There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i]. You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations. Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique.",
          "details": {
            "key_idea": "We can solve this problem using a greedy approach. The idea is to keep track of the total gas available and the current gas needed as we iterate through the stations. If the current gas becomes negative, we reset the starting station to the next station and reset the current gas needed. At the end, if the total gas available is greater than or equal to the current gas needed, then the starting station is a valid solution.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:\n        total_gas = 0\n        current_gas = 0\n        start_station = 0\n\n        for i in range(len(gas)):\n            total_gas += gas[i] - cost[i]\n            current_gas += gas[i] - cost[i]\n\n            if current_gas < 0:\n                start_station = i + 1\n                current_gas = 0\n\n        return start_station if total_gas >= 0 else -1"
          }
        },
        {
          "id": 846,
          "title": "Hand of Straights",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/hand-of-straights/",
          "description": "Alice has a hand of cards, given as an array of integers cards where cards[i] is the ith card in the hand. She wants to rearrange the cards into groups so that each group consists of k consecutive cards with ascending order. Return true if she can rearrange the cards, and false otherwise.",
          "details": {
            "key_idea": "We can solve this problem using a greedy approach. The idea is to sort the hand and use a Counter to keep track of the frequency of each card. Then, for each card, we check if there are enough consecutive cards to form a group of size W. If so, we decrement the frequencies accordingly. If not, the hand cannot be grouped and we return False.",
            "time_complexity": "O(n * log n)",
            "space_complexity": "O(n)",
            "python_solution": "from collections import Counter\n\n\nclass Solution:\n    def isNStraightHand(self, hand: List[int], W: int) -> bool:\n        if len(hand) % W != 0:\n            return False\n\n        counter = Counter(hand)\n        hand.sort()\n\n        for card in hand:\n            if counter[card] > 0:\n                for i in range(W):\n                    if counter[card + i] <= 0:\n                        return False\n                    counter[card + i] -= 1\n\n        return True"
          }
        },
        {
          "id": 763,
          "title": "Partition Labels",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/partition-labels/",
          "description": "You are given a string s. We want to partition the string into as many parts as possible so that each letter appears in at most one part. Return a list of integers representing the size of these parts.",
          "details": {
            "key_idea": "We can solve this problem using a greedy approach. The idea is to traverse the string and keep track of the last index at which each character appears. While traversing, we maintain a current partition's start and end indices. When the current index reaches the end index of the current partition, we add the length of the partition to the result list and update the start index for the next partition.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def partitionLabels(self, s: str) -> List[int]:\n        last_index = {}\n\n        for i, char in enumerate(s):\n            last_index[char] = i\n\n        result = []\n        start, end = 0, 0\n\n        for i, char in enumerate(s):\n            end = max(end, last_index[char])\n\n            if i == end:\n                result.append(end - start + 1)\n                start = end + 1\n\n        return result"
          }
        },
        {
          "id": 678,
          "title": "Valid Parenthesis String",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/valid-parenthesis-string/",
          "description": "Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.",
          "details": {
            "key_idea": "We can solve this problem using a greedy approach. The idea is to keep track of the possible lower and upper bound of valid open parentheses count as we traverse the string. For every '(', we increment both lower and upper bounds, for every ')', we decrement both bounds, and for '*', we increment the lower bound and decrement the upper bound. At each step, we ensure that the lower bound never goes below 0.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def checkValidString(self, s: str) -> bool:\n        lower = 0\n        upper = 0\n\n        for char in s:\n            if char == \"(\":\n                lower += 1\n                upper += 1\n            elif char == \")\":\n                lower = max(lower - 1, 0)\n                upper -= 1\n            else:  # char == '*'\n                lower = max(lower - 1, 0)\n                upper += 1\n\n            if upper < 0:\n                return False\n\n        return lower == 0"
          }
        }
      ]
    },
    {
      "topic_name": "Intervals",
      "problems": [
        {
          "id": 57,
          "title": "Insert Interval",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/insert-interval/",
          "description": "You are given a list of intervals where intervals[i] = [lefti, righti] represents the ith interval and an interval newInterval = [left, right] represents the interval to be inserted. Merge overlapping intervals, return intervals after the merge.",
          "details": {
            "key_idea": "We can solve this problem by iterating through the intervals and keeping track of the merged intervals. We initialize an empty list `result` to store the merged intervals. We iterate through the intervals, and for each interval, we compare its end with the start of the new interval and the new interval's end with the start of the current interval. If they overlap, we update the start and end of the new interval accordingly. If they don't overlap, we add the current interval to the result and reset the new interval. After iterating through all intervals, we add the new interval (if not added already) and return the result.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:\n        result = []\n        new_start, new_end = newInterval\n\n        for interval in intervals:\n            if interval[1] < new_start:\n                result.append(interval)\n            elif interval[0] > new_end:\n                result.append([new_start, new_end])\n                new_start, new_end = interval\n            else:\n                new_start = min(new_start, interval[0])\n                new_end = max(new_end, interval[1])\n\n        result.append([new_start, new_end])\n        return result"
          }
        },
        {
          "id": 56,
          "title": "Merge Intervals",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/merge-intervals/",
          "description": "Given a list of intervals where intervals[i] = [lefti, righti] represents the ith interval and an interval newInterval = [left, right] represents the interval to be inserted. Merge overlapping intervals, return intervals after the merge.",
          "details": {
            "key_idea": "We can solve this problem by first sorting the intervals based on their start times. Then, we iterate through the sorted intervals and merge overlapping intervals. We initialize an empty list `result` to store the merged intervals. For each interval, if it overlaps with the previous interval (i.e., the end of the previous interval is greater than or equal to the start of the current interval), we update the end of the previous interval to be the maximum of the end of both intervals. If there is no overlap, we add the previous interval to the result and update the previous interval to be the current interval. After iterating through all intervals, we add the last interval (or the merged interval) to the result.",
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def merge(self, intervals: List[List[int]]) -> List[List[int]]:\n        if not intervals:\n            return []\n\n        intervals.sort(key=lambda x: x[0])\n        result = [intervals[0]]\n\n        for interval in intervals[1:]:\n            if interval[0] <= result[-1][1]:\n                result[-1][1] = max(result[-1][1], interval[1])\n            else:\n                result.append(interval)\n\n        return result"
          }
        },
        {
          "id": 435,
          "title": "Non-overlapping Intervals",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/non-overlapping-intervals/",
          "description": "Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.",
          "details": {
            "key_idea": "We can solve this problem by first sorting the intervals based on their end times. Then, we iterate through the sorted intervals and count the number of overlapping intervals. If the start of the current interval is less than the end of the previous interval, it means there is an overlap, so we increment the count and skip adding the current interval to the non-overlapping set. If there is no overlap, we update the end of the previous interval to be the end of the current interval.",
            "time_complexity": "O(n log n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:\n        if not intervals:\n            return 0\n\n        intervals.sort(key=lambda x: x[1])\n        non_overlapping = 1  # Count of non-overlapping intervals\n        prev_end = intervals[0][1]\n\n        for i in range(1, len(intervals)):\n            if intervals[i][0] >= prev_end:\n                non_overlapping += 1\n                prev_end = intervals[i][1]\n\n        return len(intervals) - non_overlapping"
          }
        },
        {
          "id": 252,
          "title": "Meeting Rooms",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/meeting-rooms/",
          "description": "Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.",
          "details": {
            "key_idea": "We can solve this problem by sorting the intervals based on their start times and then checking for any overlapping intervals. If the start time of the current interval is less than the end time of the previous interval, it means there is an overlap, and the person cannot attend all meetings. Otherwise, there is no overlap, and the person can attend all meetings.",
            "time_complexity": "O(n log n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:\n        if not intervals:\n            return True\n\n        intervals.sort(key=lambda x: x[0])\n\n        for i in range(1, len(intervals)):\n            if intervals[i][0] < intervals[i - 1][1]:\n                return False\n\n        return True"
          }
        },
        {
          "id": 253,
          "title": "Meeting Rooms II",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/meeting-rooms-ii/",
          "description": "Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.",
          "details": {
            "key_idea": "We can solve this problem using a priority queue (min-heap). First, we sort the intervals based on their start times. We then iterate through the sorted intervals and maintain a min-heap of end times of ongoing meetings. For each interval, if the start time is greater than or equal to the smallest end time in the min-heap, it means the current meeting can reuse an existing room, so we pop the smallest end time from the min-heap. If not, we need to allocate a new room. After processing all intervals, the size of the min-heap gives us the minimum number of meeting rooms required.",
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "python_solution": "import heapq\n\n\nclass Solution:\n    def minMeetingRooms(self, intervals: List[List[int]]) -> int:\n        if not intervals:\n            return 0\n\n        intervals.sort(key=lambda x: x[0])\n        min_heap = []\n        heapq.heappush(min_heap, intervals[0][1])\n\n        for i in range(1, len(intervals)):\n            if intervals[i][0] >= min_heap[0]:\n                heapq.heappop(min_heap)\n            heapq.heappush(min_heap, intervals[i][1])\n\n        return len(min_heap)"
          }
        },
        {
          "id": 1851,
          "title": "Minimum Interval to Include Each Query",
          "difficulty": "Hard",
          "link": "https://leetcode.com/problems/minimum-interval-to-include-each-query/",
          "description": "You are given an array intervals where intervals[i] = [lefti, righti] represents the ith interval and an array queries represented by an integer array queries where queries[j] = [j]. The jth query is interested in finding the shortest interval that includes the jth query. Return an array answer where answer[j] is the shortest interval that includes the jth query. If no interval includes the jth query, answer[j] should be -1.",
          "details": {
            "key_idea": "The problem is to find the minimum interval for each query that includes at least one element from each interval in the list. We can solve this problem using a sorted list or priority queue (min-heap). First, we sort both the intervals and the queries based on their start values. Then, for each query, we iterate through the sorted intervals and add intervals to the min-heap as long as their start values are less than or equal to the current query. We also remove intervals from the min-heap whose end values are less than the start value of the current query. After processing all queries, the size of the min-heap gives us the minimum interval for each query.",
            "time_complexity": "O((n + q) log n)",
            "space_complexity": "O(n)",
            "python_solution": "import heapq\n\n\nclass Solution:\n    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:\n        intervals.sort(key=lambda x: x[0])\n        queries_sorted = sorted(enumerate(queries), key=lambda x: x[1])\n\n        min_heap = []\n        ans = [-1] * len(queries)\n        i = 0\n\n        for query_index, query in queries_sorted:\n            while i < len(intervals) and intervals[i][0] <= query:\n                start, end = intervals[i]\n                heapq.heappush(min_heap, (end - start + 1, end))\n                i += 1\n\n            while min_heap and min_heap[0][1] < query:\n                heapq.heappop(min_heap)\n\n            if min_heap:\n                ans[query_index] = min_heap[0][0]\n\n        return ans"
          }
        }
      ]
    },
    {
      "topic_name": "Math & Geometry",
      "problems": [
        {
          "id": 48,
          "title": "Rotate Image",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/rotate-image/",
          "description": "You are given an n x n 2D matrix representing an image. Rotate the image by 90 degrees (clockwise). You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.",
          "details": {
            "key_idea": "To rotate the given matrix in-place, we can perform a series of swaps. We start by swapping elements symmetrically along the diagonal, and then swap elements symmetrically along the vertical midline. This process rotates the matrix by 90 degrees clockwise.",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def rotate(self, matrix: List[List[int]]) -> None:\n        \"\"\"\n        Do not return anything, modify matrix in-place instead.\n        \"\"\"\n        n = len(matrix)\n\n        # Transpose the matrix\n        for i in range(n):\n            for j in range(i, n):\n                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]\n\n        # Reverse each row\n        for i in range(n):\n            matrix[i].reverse()"
          }
        },
        {
          "id": 54,
          "title": "Spiral Matrix",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/spiral-matrix/",
          "description": "Given an m x n matrix, return all elements of the matrix in spiral order.",
          "details": {
            "key_idea": "To traverse a matrix in a spiral order, we can iterate through each layer of the matrix and extract the elements in the desired order: top row, right column, bottom row, and left column. We update the boundaries of each layer as we traverse.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:\n        if not matrix:\n            return []\n\n        rows, cols = len(matrix), len(matrix[0])\n        result = []\n\n        # Define the boundaries of the current layer\n        top, bottom, left, right = 0, rows - 1, 0, cols - 1\n\n        while top <= bottom and left <= right:\n            # Traverse the top row\n            for j in range(left, right + 1):\n                result.append(matrix[top][j])\n            top += 1\n\n            # Traverse the right column\n            for i in range(top, bottom + 1):\n                result.append(matrix[i][right])\n            right -= 1\n\n            # Traverse the bottom row\n            if top <= bottom:  # Avoid duplicate traversal\n                for j in range(right, left - 1, -1):\n                    result.append(matrix[bottom][j])\n                bottom -= 1\n\n            # Traverse the left column\n            if left <= right:  # Avoid duplicate traversal\n                for i in range(bottom, top - 1, -1):\n                    result.append(matrix[i][left])\n                left += 1\n\n        return result"
          }
        },
        {
          "id": 73,
          "title": "Set Matrix Zeroes",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/set-matrix-zeroes/",
          "description": "Given an m x n matrix, if an element is 0, set its entire row and column to 0's. Do it in-place.",
          "details": {
            "key_idea": "To set the entire row and column to zeros if an element in the matrix is zero, we can use two sets to keep track of the rows and columns that need to be set to zero. We iterate through the matrix and mark the corresponding rows and columns in the sets. Then, we iterate through the matrix again and set the elements to zero if their row or column is marked.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m + n)",
            "python_solution": "class Solution:\n    def setZeroes(self, matrix: List[List[int]]) -> None:\n        \"\"\"\n        Do not return anything, modify matrix in-place instead.\n        \"\"\"\n        rows, cols = len(matrix), len(matrix[0])\n        zero_rows, zero_cols = set(), set()\n\n        # Mark rows and columns that need to be set to zero\n        for i in range(rows):\n            for j in range(cols):\n                if matrix[i][j] == 0:\n                    zero_rows.add(i)\n                    zero_cols.add(j)\n\n        # Set elements to zero based on marked rows and columns\n        for i in range(rows):\n            for j in range(cols):\n                if i in zero_rows or j in zero_cols:\n                    matrix[i][j] = 0"
          }
        },
        {
          "id": 202,
          "title": "Happy Number",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/happy-number/",
          "description": "A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits. Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Return true if n is a happy number, and false if not.",
          "details": {
            "key_idea": "A number is a happy number if, after repeatedly replacing it with the sum of the square of its digits, the process eventually reaches 1. To determine if a number is a happy number, we can use Floyd's Cycle Detection Algorithm. We use two pointers: a slow pointer that advances one step at a time and a fast pointer that advances two steps at a time. If the fast pointer eventually reaches the slow pointer, we have a cycle, indicating that the number is not a happy number.",
            "time_complexity": "O(log n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def isHappy(self, n: int) -> bool:\n        def get_next(num):\n            next_num = 0\n            while num > 0:\n                num, digit = divmod(num, 10)\n                next_num += digit**2\n            return next_num\n\n        slow, fast = n, get_next(n)\n\n        while fast != 1 and slow != fast:\n            slow = get_next(slow)\n            fast = get_next(get_next(fast))\n\n        return fast == 1"
          }
        },
        {
          "id": 66,
          "title": "Plus One",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/plus-one/",
          "description": "You are given a large integer represented as an array of digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's. Increment the large integer by one and return the resulting array of digits.",
          "details": {
            "key_idea": "Given a non-empty array representing a non-negative integer, we need to add one to the integer. We can perform this addition by starting from the least significant digit (the last element of the array) and moving towards the most significant digit. If the current digit is less than 9, we can simply increment it and return the modified array. Otherwise, we set the current digit to 0 and continue the process with the previous digit.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def plusOne(self, digits: List[int]) -> List[int]:\n        n = len(digits)\n\n        for i in range(n - 1, -1, -1):\n            if digits[i] < 9:\n                digits[i] += 1\n                return digits\n            digits[i] = 0\n\n        return [1] + digits"
          }
        },
        {
          "id": 50,
          "title": "Pow(x, n)",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/powx-n/",
          "description": "Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).",
          "details": {
            "key_idea": "The problem is to calculate x raised to the power n. We can solve this problem using the concept of exponentiation by squaring. If n is even, we can compute x^n as (x^(n//2)) * (x^(n//2)). If n is odd, we can compute x^n as x * (x^(n//2)) * (x^(n//2)).",
            "time_complexity": "O(log n)",
            "space_complexity": "O(log n)",
            "python_solution": "class Solution:\n    def myPow(self, x: float, n: int) -> float:\n        def helper(base, exp):\n            if exp == 0:\n                return 1.0\n            temp = helper(base, exp // 2)\n            if exp % 2 == 0:\n                return temp * temp\n            else:\n                return base * temp * temp\n\n        if n < 0:\n            x = 1 / x\n            n = -n\n\n        return helper(x, n)"
          }
        },
        {
          "id": 43,
          "title": "Multiply Strings",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/multiply-strings/",
          "description": "Given two non-negative integers num1 and num2 represented as strings, return the product of the two numbers, also represented as a string.",
          "details": {
            "key_idea": "The problem is to multiply two given non-negative integers represented as strings. We can perform multiplication digit by digit, similar to how we do multiplication manually. We start from the least significant digits and move towards the most significant digits. We use two nested loops to multiply each pair of digits and keep track of carry values.",
            "time_complexity": "O(m * n)",
            "space_complexity": "O(m + n)",
            "python_solution": "class Solution:\n    def multiply(self, num1: str, num2: str) -> str:\n        if num1 == \"0\" or num2 == \"0\":\n            return \"0\"\n\n        m, n = len(num1), len(num2)\n        result = [0] * (m + n)\n\n        for i in range(m - 1, -1, -1):\n            for j in range(n - 1, -1, -1):\n                product = int(num1[i]) * int(num2[j])\n                sum_val = product + result[i + j + 1]\n                result[i + j + 1] = sum_val % 10\n                result[i + j] += sum_val // 10\n\n        return \"\".join(map(str, result)).lstrip(\"0\")"
          }
        },
        {
          "id": 2013,
          "title": "Detect Squares",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/detect-squares/",
          "description": "You are given a stream of points with integer coordinates in the 2D plane. The data is given in the format [x, y], where x and y are integers. You are also given a stream of queries represented by the integer array queries where queries[j] = [x, y]. For each query, you want to count the number of squares that satisfy the following conditions: A square is formed by three points in the stream and a query point. The query point is the fourth point of the square. The sides of the square are parallel to the axes.",
          "details": {
            "key_idea": "The key idea is to group points by their x-coordinates and then use these groups to efficiently find potential square corners.",
            "time_complexity": "Add: O(1), Count: O(K)",
            "space_complexity": "O(N)",
            "python_solution": "from collections import defaultdict\nfrom typing import List\n\n\nclass DetectSquares:\n    def __init__(self):\n        self.points = defaultdict(lambda: defaultdict(int))\n\n    def add(self, point: List[int]) -> None:\n        x, y = point\n        self.points[x][y] += 1\n\n    def count(self, point: List[int]) -> int:\n        x, y = point\n        count = 0\n\n        for y2 in self.points[x]:\n            if y2 != y:\n                side_length = abs(y2 - y)\n\n                for x2 in (x + side_length, x - side_length):\n                    if x2 in self.points and y in self.points[x2]:\n                        count += (\n                            self.points[x2][y]\n                            * self.points[x2][y2]\n                            * self.points[x][y2]\n                        )\n\n        return count\n\n\n# Your DetectSquares object will be instantiated and called as such:\n# obj = DetectSquares()\n# obj.add(point)\n# param_2 = obj.count(point)\n"
          }
        }
      ]
    },
    {
      "topic_name": "Bit Manipulation",
      "problems": [
        {
          "id": 136,
          "title": "Single Number",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/single-number/",
          "description": "Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.",
          "details": {
            "key_idea": "To find the single number in an array where all other numbers appear twice, we can use the XOR operation. XORing a number with itself results in 0, and XORing any number with 0 results in the number itself. Therefore, if we XOR all the numbers in the array, the duplicates will cancel out, leaving only the single number.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def singleNumber(self, nums: List[int]) -> int:\n        result = 0\n        for num in nums:\n            result ^= num\n        return result"
          }
        },
        {
          "id": 191,
          "title": "Number of 1 Bits",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/number-of-1-bits/",
          "description": "Write a function that takes the binary representation of an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).",
          "details": {
            "key_idea": "To count the number of 1 bits in an unsigned integer, we can use bit manipulation. We iterate through each bit of the number and use a bitwise AND operation with 1 to check if the least significant bit is 1. If it is, we increment the count and right-shift the number by 1 to check the next bit.",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def hammingWeight(self, n: int) -> int:\n        count = 0\n        while n:\n            count += n & 1\n            n = n >> 1\n        return count"
          }
        },
        {
          "id": 338,
          "title": "Counting Bits",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/counting-bits/",
          "description": "Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.",
          "details": {
            "key_idea": "To count the number of 1 bits for each number in the range [0, num], we can use dynamic programming. We observe that the number of 1 bits in a number x is equal to the number of 1 bits in x // 2 plus the value of the least significant bit (x % 2).",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "python_solution": "class Solution:\n    def countBits(self, num: int) -> List[int]:\n        bits_count = [0] * (num + 1)\n\n        for i in range(1, num + 1):\n            bits_count[i] = bits_count[i // 2] + (i % 2)\n\n        return bits_count"
          }
        },
        {
          "id": 190,
          "title": "Reverse Bits",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/reverse-bits/",
          "description": "Write a function that takes the binary representation of an unsigned integer and returns the number that has the bits reversed.",
          "details": {
            "key_idea": "To reverse the bits of a given unsigned integer, we can iterate through each bit of the input number. For each bit, we use bitwise operations to reverse the bit and append it to the result.",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def reverseBits(self, n: int) -> int:\n        reversed_num = 0\n        for _ in range(32):\n            reversed_num = (reversed_num << 1) | (n & 1)\n            n = n >> 1\n        return reversed_num"
          }
        },
        {
          "id": 268,
          "title": "Missing Number",
          "difficulty": "Easy",
          "link": "https://leetcode.com/problems/missing-number/",
          "description": "Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.",
          "details": {
            "key_idea": "To find the missing number in an array containing distinct numbers from 0 to n, we can use the XOR operation. We XOR all the numbers from 0 to n and then XOR the result with all the elements in the array. The XOR operation cancels out the duplicate numbers, leaving only the missing number.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def missingNumber(self, nums: List[int]) -> int:\n        n = len(nums)\n        missing_num = n\n\n        for i in range(n):\n            missing_num ^= i ^ nums[i]\n\n        return missing_num"
          }
        },
        {
          "id": 371,
          "title": "Sum of Two Integers",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/sum-of-two-integers/",
          "description": "Given two integers a and b, return the sum of the two integers without using the operators + and -.",
          "details": {
            "key_idea": "To calculate the sum of two integers without using the + and - operators, we can use bitwise operations. We use the XOR operation (^) to calculate the sum without considering the carry, and the AND operation (&) followed by a left shift (<<) to calculate the carry. We repeat these steps until there is no carry left.",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def getSum(self, a: int, b: int) -> int:\n        MASK = 0xFFFFFFFF\n        MAX_INT = 0x7FFFFFFF\n\n        while b != 0:\n            a, b = (a ^ b) & MASK, ((a & b) << 1) & MASK\n\n        return a if a <= MAX_INT else ~(a ^ MASK)"
          }
        },
        {
          "id": 7,
          "title": "Reverse Integer",
          "difficulty": "Medium",
          "link": "https://leetcode.com/problems/reverse-integer/",
          "description": "Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-2^31, 2^31 - 1], then return 0.",
          "details": {
            "key_idea": "To reverse an integer, we can use integer arithmetic. We repeatedly extract the last digit of the number using the modulo operator (%) and add it to the reversed number after multiplying by 10. We then update the original number by integer division (//) to remove the last digit. We continue this process until the original number becomes 0.",
            "time_complexity": "O(log(x))",
            "space_complexity": "O(1)",
            "python_solution": "class Solution:\n    def reverse(self, x: int) -> int:\n        INT_MAX = 2**31 - 1\n        INT_MIN = -(2**31)\n\n        reversed_num = 0\n        sign = 1 if x > 0 else -1\n        x = abs(x)\n\n        while x != 0:\n            pop = x % 10\n            x //= 10\n\n            if reversed_num > (INT_MAX - pop) // 10:\n                return 0\n\n            reversed_num = reversed_num * 10 + pop\n\n        return reversed_num * sign"
          }
        }
      ]
    }
}