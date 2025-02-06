import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../../constants.dart';

class VerificationScreen extends StatefulWidget {
  final String contactInfo;
  final bool isEmail;

  const VerificationScreen({
    super.key,
    required this.contactInfo,
    required this.isEmail,
  });

  @override
  State<VerificationScreen> createState() => _VerificationScreenState();
}

class _VerificationScreenState extends State<VerificationScreen> {
  final List<TextEditingController> _controllers = List.generate(
    4,
    (index) => TextEditingController(),
  );
  final List<FocusNode> _focusNodes = List.generate(
    4,
    (index) => FocusNode(),
  );
  
  int _resendTimer = 60;
  bool _canResend = false;

  @override
  void initState() {
    super.initState();
    _startResendTimer();
  }

  @override
  void dispose() {
    for (var controller in _controllers) {
      controller.dispose();
    }
    for (var node in _focusNodes) {
      node.dispose();
    }
    super.dispose();
  }

  void _startResendTimer() {
    setState(() {
      _resendTimer = 60;
      _canResend = false;
    });
    Future.delayed(const Duration(seconds: 1), () {
      if (mounted && _resendTimer > 0) {
        setState(() {
          _resendTimer--;
        });
        if (_resendTimer > 0) {
          _startResendTimer();
        } else {
          setState(() {
            _canResend = true;
          });
        }
      }
    });
  }

  void _onCodeChanged(String value, int index) {
    if (value.length == 1 && index < 3) {
      _focusNodes[index + 1].requestFocus();
    }
    
    // Check if all fields are filled
    if (_controllers.every((controller) => controller.text.isNotEmpty)) {
      // TODO: Implement verification logic
      String code = _controllers.map((controller) => controller.text).join();
      print('Verification code entered: $code');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.white),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppTheme.spacing_lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Enter verification code',
                style: TextStyle(
                  color: AppColors.white,
                  fontSize: AppTheme.fontSize_xl,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              Text(
                'We sent a code to ${widget.isEmail ? "email" : "phone number"}\n${widget.contactInfo}',
                style: const TextStyle(
                  color: AppColors.grey,
                  fontSize: AppTheme.fontSize_md,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl * 2),
              // Code input boxes
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: List.generate(
                  4,
                  (index) => SizedBox(
                    width: 60,
                    height: 60,
                    child: TextField(
                      controller: _controllers[index],
                      focusNode: _focusNodes[index],
                      keyboardType: TextInputType.number,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: AppColors.white,
                        fontSize: AppTheme.fontSize_xl,
                        fontWeight: FontWeight.bold,
                      ),
                      decoration: InputDecoration(
                        filled: true,
                        fillColor: AppColors.darkGrey,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                          borderSide: BorderSide.none,
                        ),
                      ),
                      inputFormatters: [
                        LengthLimitingTextInputFormatter(1),
                        FilteringTextInputFormatter.digitsOnly,
                      ],
                      onChanged: (value) => _onCodeChanged(value, index),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl * 2),
              // Resend timer/button
              Center(
                child: _canResend
                    ? TextButton(
                        onPressed: () {
                          // TODO: Implement resend logic
                          _startResendTimer();
                        },
                        child: const Text(
                          'Send new code',
                          style: TextStyle(
                            color: AppColors.accent,
                            fontSize: AppTheme.fontSize_md,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      )
                    : Text(
                        'Send new code (${_resendTimer}s)',
                        style: const TextStyle(
                          color: AppColors.grey,
                          fontSize: AppTheme.fontSize_md,
                        ),
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
