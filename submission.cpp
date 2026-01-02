#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, K, B;
    cin >> N >> M >> K >> B;
    vector<int> min_range, max_range;
    int input1, input2, min_val;

    for(int i = 0; i < N; i++){
        cin >> input1 >> input2;
        min_val = abs(input1 - input2) + B;
        min_range.push_back(input1 + input2 - min_val);
        max_range.push_back(input1 + input2 + min_val);
    }

    auto can = [&](int T){
        vector<pair<int,int>> total_ranges;
        for(int i : min_range){
            total_ranges.push_back({max(0,(i-T)/2),1});
        }
        for(int i : max_range){
            total_ranges.push_back({min(K-1,(i+T)/2+1),-1});
        }
        sort(total_ranges.begin(),total_ranges.end());
        int passed = 0;
        for(auto i : total_ranges){
            passed += i.second;
            if (passed >= M){
                return true;
            }
        }
        return false;
    };

    int left = 0;
    int right = 2*K + 1 - B;
    while(right >= left){
        int mid = (left+right)/2;
        if(can(mid)){
            right = mid - 1;
        }
        else{
            left = mid + 1;
        }
    }
    cout << left;
}