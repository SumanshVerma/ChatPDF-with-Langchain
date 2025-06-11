css = '''
<style>
.question-container {
    background-color: #333;
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    font-size: 18px;
}
.answer-container {
    background-color: #222;
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    font-size: 16px;
    line-height: 1.5;
}
</style>
'''

bot_template = '''
<div class="answer-container">
    {{MSG}}
</div>
'''


user_template = '''
<div class="question-container">
    {{MSG}}
</div>
'''